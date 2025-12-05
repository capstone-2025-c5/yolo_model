# run_model_live.py
# Capture → YOLO → display (+ Protobuf/MCAP emission or Foxglove streaming)

import time
import argparse
import cv2
import os
import numpy as np
import logging
from typing import Iterable, Tuple, Optional, Dict, List
from threading import Thread, Event
from queue import Queue, Empty
import traceback

import config as C
from frame_source import (
    GStreamerWindowFrameSource,
    OpenCVCameraFrameSource,
    # DXCamWindowFrameSource,
    MCAPFoxgloveCompressedVideoSource,
)
from yolo_infer import YOLORunner
from third_party.protos_py import plane_detection_pb2 as pd_pb2
from adsb_geojson_streamer import ADSBDataReader, GeoJSONStreamer, time_sync_adsb_to_video
from alert_manager import AlertManager
from alert_manager import AlertManager

# --- Optional MCAP support (only if --out-mcap is provided) ---
_MCAP_AVAILABLE = True
try:
    # from mcap.mcap0.writer import Writer as McapWriter  # type: ignore
    from google.protobuf import descriptor_pb2  # for schema FileDescriptorSet
except Exception:
    _MCAP_AVAILABLE = False

try:
    import av  # PyAV for H.264/H.265 encoding
    _PYAV_AVAILABLE = True
except Exception:
    _PYAV_AVAILABLE = False

# --- Foxglove streaming support ---
_FOXGLOVE_AVAILABLE = True
try:
    import foxglove
    from foxglove.channels import RawImageChannel
    from foxglove.schemas import RawImage, Timestamp
except Exception:
    _FOXGLOVE_AVAILABLE = False


def make_source(args, topic=None):
    """Create a frame source. If topic is specified, override args.topic for MCAP sources."""
    if args.source == "gst_window":
        return GStreamerWindowFrameSource(C.TARGET_TITLE_CONTAINS, fps=C.TARGET_FPS)
    elif args.source == "dx_window":
        raise RuntimeError("DXCam not available on this system")
    elif args.source == "camera":
        return OpenCVCameraFrameSource(index=args.camera_index, width=C.TARGET_W, height=C.TARGET_H, fps=C.TARGET_FPS)
    elif args.source == "mcap":
        if not args.mcap:
            raise ValueError("When --source mcap, you must provide --mcap <path to .mcap>")
        return MCAPFoxgloveCompressedVideoSource(
            mcap_path=args.mcap,
            topic=topic or args.topic,
            realtime=(not args.no_realtime),
            start_offset_s=args.start_offset,
            start_abs_ns=args.start_abs_ns,
            preroll_s=args.preroll,
        )
    else:
        raise ValueError(f"Unknown source: {args.source}")


# -------------------------------
# YOLO result → best airplane
# -------------------------------
def _best_airplane_detection(
    results,
    frame_shape: Tuple[int, int, int],
    preferred_names: Iterable[str],
) -> Tuple[bool, Optional[Tuple[float, float, float, float, float]]]:
    """
    Returns (present, (cx, cy, bw, bh, conf)) with normalized coords, or (False, None).
    Works with typical Ultralytics YOLOv8 Results object produced by your YOLORunner.
    """
    preferred = set(n.lower() for n in preferred_names)
    try:
        r = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(r, "boxes", None)
        names = getattr(r, "names", None)
        if boxes is None or names is None:
            return False, None

        H, W = frame_shape[0], frame_shape[1]
        best = None  # (cx, cy, bw, bh, conf)
        n = len(boxes)
        for i in range(n):
            cls_id = int(boxes.cls[i].item())
            name = names.get(cls_id, "").lower()
            if name not in preferred:
                continue
            conf = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            cx = ((x1 + x2) / 2.0) / W
            cy = ((y1 + y2) / 2.0) / H
            bw = (x2 - x1) / W
            bh = (y2 - y1) / H
            if best is None or conf > best[-1]:
                best = (cx, cy, bw, bh, conf)

        if best is None:
            return False, None
        return True, best
    except Exception:
        # Be conservative on parsing errors
        return False, None


# --------------------------------------
# Build PlaneState protobuf per frame
# --------------------------------------
_frame_seq = 0


def build_plane_state(
    frame,
    results,
    class_filter: Optional[Iterable[str]],
    source_topic: str = "",
    source_file: str = "",
) -> pd_pb2.PlaneState:
    global _frame_seq
    _frame_seq += 1
    H, W = frame.shape[:2]
    preferred = class_filter if class_filter else ["airplane"]

    present, best = _best_airplane_detection(results, frame.shape, preferred)

    msg = pd_pb2.PlaneState()
    msg.plane_present = bool(present)
    msg.timestamp_unix_ns = int(time.time_ns())
    msg.frame_seq = int(_frame_seq)
    if source_topic:
        msg.source_topic = source_topic
    if source_file:
        msg.source_file = source_file
    msg.image_width = int(W)
    msg.image_height = int(H)

    if present and best is not None:
        cx, cy, bw, bh, conf = best
        msg.plane.x = float(cx)
        msg.plane.y = float(cy)
        msg.plane.confidence = float(conf)
        msg.plane.width = float(bw)
        msg.plane.height = float(bh)

    return msg


# --------------------------
# Binary .pb stream writer
# --------------------------
class PBStreamWriter:
    """
    Simple length-prefixed binary stream of protobuf messages:
      [uint32_be length][message_bytes]...
    """
    def __init__(self, path: str):
        self.path = path
        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        self.f = open(path, "ab")

    def write(self, msg) -> None:
        data = msg.SerializeToString()
        n = len(data)
        self.f.write(n.to_bytes(4, "big"))
        self.f.write(data)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass


# --------------------------
# MCAP helpers (optional)
# --------------------------
def _make_fds_for_module(mod) -> Optional["descriptor_pb2.FileDescriptorSet"]:
    """Collect FileDescriptorSet for the generated module (so MCAP consumers can decode)."""
    if not _MCAP_AVAILABLE:
        return None
    fds = descriptor_pb2.FileDescriptorSet()
    seen = set()

    def rec(fd):
        if fd.name in seen:
            return
        seen.add(fd.name)
        for dep in fd.dependencies:
            rec(dep)
        # Rehydrate a FileDescriptorProto from the serialized_pb
        proto = descriptor_pb2.FileDescriptorProto()
        proto.ParseFromString(fd.serialized_pb)
        fds.file.append(proto)

    rec(mod.DESCRIPTOR.file)
    return fds


class MCAPPlaneWriter:
    def __init__(self, path: str, topic_out: str, proto_module):
        if not _MCAP_AVAILABLE:
            raise RuntimeError("mcap is not available; install `pip install mcap` to use --out-mcap")

        dirn = os.path.dirname(path)
        if dirn:
            os.makedirs(dirn, exist_ok=True)
        self.fp = open(path, "wb")
        self.writer = McapWriter(self.fp)

        # Build schema from the generated module
        fds = _make_fds_for_module(proto_module)
        full_name = proto_module.PlaneState.DESCRIPTOR.full_name or "PlaneState"

        self.schema_id = self.writer.register_schema(
            name=full_name,  # e.g., "aircraft.detect.v1.PlaneState" if package present
            encoding="protobuf",
            data=(fds.SerializeToString() if fds is not None else b""),
        )
        self.channel_id = self.writer.register_channel(
            topic=topic_out,
            message_encoding="protobuf",
            schema_id=self.schema_id,
        )

    def write(self, msg: pd_pb2.PlaneState) -> None:
        t = time.time_ns()
        self.writer.add_message(
            channel_id=self.channel_id,
            log_time=t,
            publish_time=t,
            data=msg.SerializeToString(),
        )

    def close(self):
        try:
            self.writer.finish()
        finally:
            try:
                self.fp.close()
            except Exception:
                pass


class MCAPCompressedVideoWriter:
    """Write annotated frames as Foxglove CompressedVideo to MCAP output."""
    def __init__(self, mcap_writer, topic: str, format: str = "jpeg"):
        """
        Args:
            mcap_writer: Shared McapWriter instance
            topic: Output topic name (e.g., "/detect/camera_0/detections")
            format: "jpeg" or "h264"
        """
        self.mcap_writer = mcap_writer
        self.topic = topic
        self.format = format
        self.channel_id = None
        self._encoder = None
        self._frame_count = 0
        
        # Register schema (foxglove.CompressedVideo protobuf)
        self._register_schema()

    def _register_schema(self):
        """Register Foxglove CompressedVideo schema."""
        # Minimal schema registration for foxglove.CompressedVideo
        # In a real scenario, you'd load the full FileDescriptorSet
        schema_id = self.mcap_writer.register_schema(
            name="foxglove.CompressedVideo",
            encoding="protobuf",
            data=b"",  # For Foxglove compatibility, empty or minimal descriptor
        )
        self.channel_id = self.mcap_writer.register_channel(
            topic=self.topic,
            message_encoding="protobuf",
            schema_id=schema_id,
        )

    def write(self, frame: np.ndarray, log_time_ns: Optional[int] = None) -> None:
        """
        Encode frame and write to MCAP.
        Args:
            frame: BGR image (numpy array, uint8)
            log_time_ns: Message timestamp; if None, uses current time
        """
        if frame is None:
            return
        
        if log_time_ns is None:
            log_time_ns = time.time_ns()

        try:
            if self.format == "jpeg":
                # JPEG encoding (simpler, more universally supported)
                success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    return
                payload = bytes(buffer)
            else:
                # H.264 encoding (requires PyAV)
                if not _PYAV_AVAILABLE:
                    raise RuntimeError("PyAV required for H.264 encoding")
                payload = self._encode_h264(frame)
                if payload is None:
                    return

            # Build Foxglove CompressedVideo message
            # This is a minimal protobuf that Foxglove can deserialize
            msg_data = self._build_compressed_video_protobuf(payload)
            
            self.mcap_writer.add_message(
                channel_id=self.channel_id,
                log_time=log_time_ns,
                publish_time=log_time_ns,
                data=msg_data,
            )
        except Exception as e:
            print(f"Warning: Failed to write frame to MCAP: {e}")

    def _encode_h264(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode BGR frame as H.264 NAL units with keyframes."""
        if self._encoder is None:
            from fractions import Fraction
            h, w = frame.shape[:2]
            self._encoder = av.codec.CodecContext.create("h264", "w")
            self._encoder.width = w
            self._encoder.height = h
            self._encoder.time_base = Fraction(1, 30)  # 30 fps
            # Configure encoder for streaming with regular keyframes
            self._encoder.options = {
                "preset": "ultrafast",  # Speed over quality
                "crf": "23",  # Quality level
                "g": "30",  # Keyframe interval (gop size) - keyframe every 30 frames
                "forced-idr": "1",  # Force IDR (keyframe) frames
            }
            self._keyframe_interval = 30
        
        try:
            av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            # Request keyframe every N frames
            if self._frame_count % self._keyframe_interval == 0:
                av_frame.key_frame = True
            
            packets = self._encoder.encode(av_frame)
            self._frame_count += 1
            if packets:
                return b"".join(p.to_bytes() for p in packets)
        except Exception as e:
            logging.warning(f"H.264 encoding error in MCAP: {e}")
        return None

    @staticmethod
    def _build_compressed_video_protobuf(payload: bytes) -> bytes:
        """Build a minimal foxglove.CompressedVideo protobuf message."""
        # Foxglove CompressedVideo schema (simplified):
        # message CompressedVideo {
        #   int64 timestamp = 1;
        #   string format = 2;
        #   bytes data = 3;
        # }
        # Using manual protobuf encoding (field 2=string "jpeg", field 3=bytes)
        msg = bytearray()
        
        # Field 2: format (string "jpeg" or "h264") - wire type 2 (length-delimited)
        fmt = b"jpeg"
        msg.append((2 << 3) | 2)  # field 2, wire type 2
        msg.append(len(fmt))
        msg.extend(fmt)
        
        # Field 3: data (bytes) - wire type 2 (length-delimited)
        msg.append((3 << 3) | 2)  # field 3, wire type 2
        # Encode length as varint
        n = len(payload)
        while n >= 128:
            msg.append((n & 0x7f) | 0x80)
            n >>= 7
        msg.append(n & 0x7f)
        msg.extend(payload)
        
        return bytes(msg)


class FoxgloveRawImageWriter:
    """Stream annotated frames as Foxglove RawImage over WebSocket."""
    def __init__(self, topic: str, context):
        """
        Args:
            topic: Output topic name (e.g., "/detect/camera_0/detections")
            context: Foxglove context (same as server context)
        """
        self.topic = topic
        self.context = context
        self._channel = None
        self._initialize_channel()
        
    def _initialize_channel(self) -> None:
        """Create RawImage channel with the provided context."""
        if self._channel is None and _FOXGLOVE_AVAILABLE and self.context is not None:
            try:
                self._channel = RawImageChannel(
                    topic=self.topic,
                    context=self.context
                )
                logging.info(f"Initialized Foxglove RawImage channel: {self.topic}")
            except Exception as e:
                logging.error(f"Failed to create RawImageChannel: {e}")

    def write(self, frame: np.ndarray) -> None:
        """
        Send frame as RawImage to Foxglove.
        Args:
            frame: BGR image (numpy array, uint8)
        """
        if frame is None or self._channel is None:
            return

        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = rgb_frame.shape[:2]
            
            # Create RawImage message
            msg = RawImage(
                timestamp=Timestamp.from_epoch_secs(time.time()),
                data=rgb_frame.tobytes(),
                width=w,
                height=h,
                encoding="rgb8",
                step=w * 3,  # bytes per row for RGB8
                frame_id=f"frame_{int(time.time() * 1000)}"
            )
            
            self._channel.log(msg)
        except Exception as e:
            logging.warning(f"Failed to write frame to Foxglove: {e}")


class FoxgloveConfidenceWriter:
    """Stream detection confidence as a scalar value to Foxglove."""
    def __init__(self, topic: str, context):
        """
        Args:
            topic: Output topic name (e.g., "/detect/camera_0/confidence")
            context: Foxglove context (same as server context)
        """
        self.topic = topic
        self.context = context
        self._channel = None
        self._initialize_channel()
        
    def _initialize_channel(self) -> None:
        """Create a generic channel for scalar confidence values."""
        if self._channel is None and _FOXGLOVE_AVAILABLE and self.context is not None:
            try:
                from foxglove import Channel, Schema
                import json
                # Simple scalar value schema
                schema_def = {
                    "type": "object",
                    "properties": {
                        "timestamp": {"type": "number"},
                        "value": {"type": "number"},
                        "label": {"type": "string"},
                    }
                }
                self._channel = Channel(
                    topic=self.topic,
                    message_encoding="json",
                    schema=Schema(
                        name="scalar",
                        encoding="jsonschema",
                        data=json.dumps(schema_def).encode("utf-8"),
                    ),
                    context=self.context,
                )
                logging.info(f"Initialized Foxglove confidence channel: {self.topic}")
            except Exception as e:
                logging.error(f"Failed to create confidence channel: {e}")

    def write(self, confidence: Optional[float], label: str = "airplane") -> None:
        """
        Stream confidence value to Foxglove.
        Args:
            confidence: Confidence score (0-1), or None if no detection
            label: Class name being detected
        """
        if self._channel is None:
            return

        try:
            msg = {
                "timestamp": time.time(),
                "value": float(confidence) if confidence is not None else 0.0,
                "label": label,
            }
            self._channel.log(msg)
        except Exception as e:
            logging.warning(f"Failed to write confidence to Foxglove: {e}")


class FoxgloveCompressedVideoWriter:
    """Stream annotated frames as Foxglove CompressedVideo over WebSocket (deprecated - use RawImageWriter)."""
    def __init__(self, topic: str, context, format: str = "jpeg"):
        """
        Args:
            topic: Output topic name (e.g., "/detect/camera_0/detections")
            context: Foxglove context (same as server context)
            format: "jpeg" or "h264"
        """
        self.topic = topic
        self.context = context
        self.format = format
        self._encoder = None
        self._frame_count = 0
        self._channel = None
        self._initialize_channel()
        
    def _initialize_channel(self) -> None:
        """Create CompressedVideo channel with the provided context."""
        if self._channel is None and _FOXGLOVE_AVAILABLE and self.context is not None:
            try:
                from foxglove.channels import CompressedVideoChannel
                self._channel = CompressedVideoChannel(
                    topic=self.topic,
                    context=self.context
                )
                logging.info(f"Initialized Foxglove CompressedVideo channel: {self.topic}")
            except Exception as e:
                logging.warning(f"Failed to create CompressedVideoChannel: {e}. Falling back to generic Channel.")
                # Fallback if CompressedVideoChannel not available
                try:
                    from foxglove import Channel, Schema
                    import json
                    schema_def = {
                        "type": "object",
                        "properties": {
                            "format": {"type": "string"},
                            "data": {"type": "string"},
                        }
                    }
                    self._channel = Channel(
                        topic=self.topic,
                        message_encoding="json",
                        schema=Schema(
                            name="foxglove.CompressedVideo",
                            encoding="jsonschema",
                            data=json.dumps(schema_def).encode("utf-8"),
                        ),
                    )
                except Exception as e2:
                    logging.error(f"Failed to create fallback Channel: {e2}")

    def write(self, frame: np.ndarray) -> None:
        """
        Encode frame and stream to Foxglove.
        Args:
            frame: BGR image (numpy array, uint8)
        """
        if frame is None or self._channel is None:
            return

        try:
            if self.format == "jpeg":
                success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not success:
                    return
                payload = bytes(buffer)
            else:
                if not _PYAV_AVAILABLE:
                    logging.warning("PyAV required for H.264 encoding, falling back to JPEG")
                    success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not success:
                        return
                    payload = bytes(buffer)
                else:
                    payload = self._encode_h264(frame)
                    if payload is None:
                        return

            # Create CompressedVideo message using proper schema
            try:
                from foxglove.schemas import CompressedVideo, Timestamp
                msg = CompressedVideo(
                    timestamp=Timestamp.from_epoch_secs(time.time()),
                    format=self.format,
                    data=payload
                )
            except ImportError:
                # Fallback: create dict message if CompressedVideo schema not available
                msg = {
                    "format": self.format,
                    "data": payload
                }
            
            self._channel.log(msg)
        except Exception as e:
            logging.warning(f"Failed to write frame to Foxglove: {e}")

    def _encode_h264(self, frame: np.ndarray) -> Optional[bytes]:
        """Encode BGR frame as H.264 NAL units with keyframes."""
        if self._encoder is None:
            from fractions import Fraction
            h, w = frame.shape[:2]
            self._encoder = av.codec.CodecContext.create("h264", "w")
            self._encoder.width = w
            self._encoder.height = h
            self._encoder.time_base = Fraction(1, 30)  # 30 fps
            # Configure encoder for streaming with regular keyframes
            self._encoder.options = {
                "preset": "ultrafast",  # Speed over quality
                "crf": "23",  # Quality level
                "g": "30",  # Keyframe interval (gop size) - keyframe every 30 frames
                "forced-idr": "1",  # Force IDR (keyframe) frames
            }
            self._keyframe_interval = 30
        
        try:
            av_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
            # Request keyframe every N frames
            if self._frame_count % self._keyframe_interval == 0:
                av_frame.key_frame = True
            
            packets = self._encoder.encode(av_frame)
            self._frame_count += 1
            if packets:
                return b"".join(p.to_bytes() for p in packets)
        except Exception as e:
            logging.warning(f"H.264 encoding error in Foxglove: {e}")
        return None


def main():
    ap = argparse.ArgumentParser(description="Live YOLO inference from pluggable sources + protobuf/video/Foxglove streaming")
    ap.add_argument("--source", default=C.DEFAULT_SOURCE, choices=["gst_window", "dx_window", "camera", "mcap"])
    ap.add_argument("--camera-index", type=int, default=0)

    # MCAP input (video) args
    ap.add_argument("--mcap", default=None)
    ap.add_argument("--topic", default=None, help="Single input topic (or use --topics for multiple)")
    ap.add_argument("--topics", nargs="*", default=None, help="Multiple input topics for parallel processing")
    ap.add_argument("--no-realtime", action="store_true")
    ap.add_argument("--start-offset", type=float, default=None)
    ap.add_argument("--start-abs-ns", type=int, default=None)
    ap.add_argument("--preroll", type=float, default=0.5)

    # YOLO args
    ap.add_argument("--weights", default=C.YOLO_WEIGHTS)
    ap.add_argument("--conf", type=float, default=C.YOLO_CONF)
    ap.add_argument("--imgsz", type=int, default=C.YOLO_IMGSZ)
    ap.add_argument("--show-fps", dest="show_fps", action="store_true")
    ap.add_argument("--class-filter", nargs="*", default=None, help="e.g. --class-filter airplane")

    # Protobuf output
    ap.add_argument("--out-pb", default=None, help="Append length-prefixed PlaneState messages to this .pb stream")
    ap.add_argument("--out-mcap", default=None, help="Write PlaneState messages and annotated video to this MCAP file")
    ap.add_argument("--topic-out", default="/detect/plane_state", help="Output topic name for PlaneState detections")
    ap.add_argument("--video-format", choices=["jpeg", "h264"], default="h264", help="Video compression format (h264 recommended for Foxglove)")

    # Foxglove streaming args
    ap.add_argument("--foxglove-host", default=None, help="Foxglove server host (e.g., localhost or jetson IP)")
    ap.add_argument("--foxglove-port", type=int, default=8765, help="Foxglove server WebSocket port")

    # ADS-B streaming args
    ap.add_argument("--adsb-mcap", default=None, help="ADS-B aircraft MCAP file to sync and stream as GeoJSON")
    ap.add_argument("--adsb-geojson-topic", default="/aircraft/positions", help="Foxglove topic for GeoJSON aircraft positions")
    
    # Alert manager args
    ap.add_argument("--enable-alerts", action="store_true", help="Enable aircraft alert system (ADS-B + YOLO)")
    ap.add_argument("--alert-polygon", type=str, default=None, help="Final approach polygon as lat,lon pairs: 'lat1,lon1;lat2,lon2;...'")
    ap.add_argument("--alert-yolo-conf", type=float, default=0.5, help="YOLO confidence threshold for alerts (default: 0.5)")
    ap.add_argument("--alert-yolo-duration", type=float, default=0.5, help="YOLO sustained detection duration in seconds (default: 0.5)")
    ap.add_argument("--alert-debounce", type=float, default=2.0, help="Alert debounce duration in seconds (default: 2.0)")
    ap.add_argument("--alert-topic", default="/aircraft/alert", help="Foxglove topic for alert state")
    ap.add_argument("--alert-gpio-pin", type=int, default=30, help="GPIO pin number for alert output (BOARD mode, default: 30)")

    # Display toggle
    ap.add_argument("--no-display", action="store_true", help="Disable display window for headless runs")

    args = ap.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Determine input topics
    if args.topics:
        input_topics = args.topics
    elif args.topic:
        input_topics = [args.topic]
    else:
        input_topics = None

    # Ensure output dirs exist if provided
    if args.out_pb:
        d = os.path.dirname(args.out_pb)
        if d:
            os.makedirs(d, exist_ok=True)
    if args.out_mcap:
        d = os.path.dirname(args.out_mcap)
        if d:
            os.makedirs(d, exist_ok=True)

    # Initialize Foxglove server if requested
    foxglove_server = None
    ctx = foxglove.Context()
    if args.foxglove_host:
        if not _FOXGLOVE_AVAILABLE:
            raise RuntimeError("foxglove SDK not available; install `pip install foxglove-sdk`")
        try:
            foxglove_server = foxglove.start_server(
                context=ctx,
                host=args.foxglove_host,
                port=args.foxglove_port,
                capabilities=[],
            )
            logging.info(f"Foxglove server started at ws://{args.foxglove_host}:{args.foxglove_port}")
        except Exception as e:
            logging.error(f"Failed to start Foxglove server: {e}")
            raise

    # Load ADS-B data if provided
    adsb_reader = None
    geojson_streamer = None
    # Alert manager initialization
    alert_manager = None
    
    if args.adsb_mcap and args.foxglove_host:
        try:
            logging.info(f"Loading ADS-B data from {args.adsb_mcap}")
            adsb_reader = ADSBDataReader(args.adsb_mcap)
            geojson_streamer = GeoJSONStreamer(topic=args.adsb_geojson_topic, context=ctx)
            
            # Initialize alert manager if enabled
            if args.enable_alerts:
                if not args.alert_polygon:
                    logging.error("--enable-alerts requires --alert-polygon to be specified")
                    return
                
                # Parse polygon coordinates
                try:
                    polygon_points = []
                    for pair in args.alert_polygon.split(';'):
                        lat, lon = map(float, pair.split(','))
                        polygon_points.append((lat, lon))
                    
                    if len(polygon_points) < 3:
                        logging.error("Alert polygon must have at least 3 points")
                        return
                    
                    alert_manager = AlertManager(
                        final_approach_polygon=polygon_points,
                        yolo_confidence_threshold=args.alert_yolo_conf,
                        yolo_duration_threshold=args.alert_yolo_duration,
                        debounce_duration=args.alert_debounce,
                        gpio_pin=args.alert_gpio_pin,
                        foxglove_context=ctx,
                        alert_topic=args.alert_topic
                    )
                    logging.info(f"Alert manager initialized with {len(polygon_points)} polygon points")
                except Exception as e:
                    logging.error(f"Failed to initialize alert manager: {e}")
                    return
            
            # Time sync ADS-B to video
            if args.mcap:
                time_sync_adsb_to_video(
                    args.adsb_mcap,
                    args.mcap,
                    video_start_offset_secs=args.start_offset or 0,
                )
            logging.info("ADS-B data loaded and ready for streaming")
        except Exception as e:
            logging.warning(f"Failed to load ADS-B data: {e}")
            adsb_reader = None
            geojson_streamer = None

    # Initialize YOLO model once (shared across threads)
    yolo = YOLORunner(weights=args.weights, conf=args.conf, imgsz=args.imgsz, allowed_names=args.class_filter)

    # Initialize output writers
    pb_writer = PBStreamWriter(args.out_pb) if args.out_pb else None
    
    mcap_writer_obj = None
    mcap_fp = None
    plane_writer = None
    video_writers = {}
    foxglove_video_writers = {}


    
    if args.out_mcap:
        mcap_fp = open(args.out_mcap, "wb")
        mcap_writer_obj = foxglove.open_mcap(args.out_mcap, context=ctx)
        
        # Register PlaneState schema
        fds = _make_fds_for_module(pd_pb2)
        plane_schema_id = mcap_writer_obj.register_schema(
            name="PlaneState",
            encoding="protobuf",
            data=(fds.SerializeToString() if fds is not None else b""),
        )
        plane_channel_id = mcap_writer_obj.register_channel(
            topic=args.topic_out,
            message_encoding="protobuf",
            schema_id=plane_schema_id,
        )
        
        plane_writer = (mcap_writer_obj, plane_channel_id)
        
        # Create video writers for each input topic
        if input_topics:
            for i, in_topic in enumerate(input_topics):
                # Generate output topic name from input topic
                out_video_topic = f"{in_topic.rstrip('/')}/detections" if in_topic else f"/detect/camera_{i}/detections"
                video_writers[in_topic] = MCAPCompressedVideoWriter(
                    mcap_writer_obj, out_video_topic, format=args.video_format
                )

    # Create Foxglove video writers if streaming
    if foxglove_server and input_topics:
        for i, in_topic in enumerate(input_topics):
            out_video_topic = f"{in_topic.rstrip('/')}/detections" if in_topic else f"/detect/camera_{i}/detections"
            foxglove_video_writers[in_topic] = FoxgloveRawImageWriter(
                out_video_topic, context=ctx
            )
            logging.info(f"Foxglove RawImage writer created for {in_topic} -> {out_video_topic}")

    # Create Foxglove confidence writers if streaming
    foxglove_confidence_writers = {}
    if foxglove_server and input_topics:
        for i, in_topic in enumerate(input_topics):
            conf_topic = f"{in_topic.rstrip('/')}/confidence" if in_topic else f"/detect/camera_{i}/confidence"
            foxglove_confidence_writers[in_topic] = FoxgloveConfidenceWriter(
                conf_topic, context=ctx
            )
            logging.info(f"Foxglove confidence writer created for {in_topic} -> {conf_topic}")

    # Single-topic processing (original behavior)
    if not input_topics:
        print("Processing single topic (no multi-camera mode)")
        src = make_source(args)
        print("Source type:", type(src).__name__)
        if not src.open():
            raise RuntimeError("Failed to open frame source")

        try:
            _process_single_source(
                src, yolo, args, pb_writer, plane_writer, video_writers, mcap_writer_obj
            )
        finally:
            src.release()

    # Multi-topic processing (new behavior)
    else:
        print(f"Processing {len(input_topics)} topics in parallel: {input_topics}")
        try:
            _process_multi_source(
                args, input_topics, yolo, pb_writer, plane_writer, video_writers, 
                foxglove_video_writers, foxglove_confidence_writers, mcap_writer_obj,
                adsb_reader, geojson_streamer, alert_manager
            )
        except KeyboardInterrupt:
            logging.info("Interrupted by user")

    # Cleanup
    if pb_writer:
        pb_writer.close()
    if mcap_writer_obj:
        mcap_writer_obj.finish()
    if mcap_fp:
        mcap_fp.close()
    if alert_manager:
        alert_manager.cleanup()
    if foxglove_server:
        try:
            foxglove_server.stop()
            logging.info("Foxglove server stopped")
        except Exception:
            pass


def _process_single_source(src, yolo, args, pb_writer, plane_writer, video_writers, mcap_writer_obj):
    """Process a single frame source (original behavior)."""
    prev, frames, fps = time.time(), 0, 0.0

    try:
        while True:
            ok, frame = src.read()
            if not ok:
                break
            if frame is None:
                continue

            results = yolo.infer(frame)
            out = yolo.draw(frame, results)

            # Build protobuf
            src_topic = args.topic or ""
            src_file = args.mcap or ""
            state = build_plane_state(
                frame=frame,
                results=results,
                class_filter=args.class_filter,
                source_topic=src_topic,
                source_file=src_file,
            )

            # Emit protobuf detections
            if pb_writer:
                pb_writer.write(state)
            if plane_writer:
                mcap_writer_obj, channel_id = plane_writer
                t = time.time_ns()
                mcap_writer_obj.add_message(
                    channel_id=channel_id,
                    log_time=t,
                    publish_time=t,
                    data=state.SerializeToString(),
                )

            # Emit annotated video
            if src_topic in video_writers:
                video_writers[src_topic].write(out)

            # FPS + display (optional)
            frames += 1
            now = time.time()
            if now - prev >= 1.0:
                fps = frames / (now - prev)
                frames, prev = 0, now

            if not args.no_display:
                if args.show_fps:
                    cv2.putText(out, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                cv2.imshow("YOLO Live", out)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        if not args.no_display:
            cv2.destroyAllWindows()


def _process_multi_source(args, input_topics, yolo, pb_writer, plane_writer, video_writers, foxglove_video_writers, foxglove_confidence_writers, mcap_writer_obj, adsb_reader=None, geojson_streamer=None, alert_manager=None):
    """Process multiple frame sources in parallel threads."""
    sources = {}
    threads = {}
    stop_event = Event()
    
    try:
        # Create and open all sources
        for topic in input_topics:
            try:
                src = make_source(args, topic=topic)
                print(f"Opening source for topic {topic}: {type(src).__name__}")
                if not src.open():
                    print(f"Warning: Failed to open source for topic {topic}")
                    continue
                sources[topic] = src
            except Exception as e:
                print(f"Warning: Failed to create source for topic {topic}: {e}")
                traceback.print_exc()
                continue

        # Start worker threads
        for topic in sources:
            t = Thread(
                target=_worker_process_topic,
                args=(topic, sources[topic], yolo, args, pb_writer, plane_writer, 
                      video_writers, foxglove_video_writers, foxglove_confidence_writers, mcap_writer_obj, stop_event, adsb_reader, geojson_streamer, alert_manager),
                daemon=False
            )
            t.start()
            threads[topic] = t

        # Wait for all threads to complete
        for topic, t in threads.items():
            t.join()
            print(f"Thread for {topic} completed")

    except KeyboardInterrupt:
        print("Interrupted, stopping all threads...")
        stop_event.set()
        for t in threads.values():
            t.join(timeout=2.0)

    finally:
        # Cleanup
        for topic, src in sources.items():
            try:
                src.release()
            except Exception:
                pass
        if not args.no_display:
            cv2.destroyAllWindows()


def _worker_process_topic(topic, src, yolo, args, pb_writer, plane_writer, video_writers, foxglove_video_writers, foxglove_confidence_writers, mcap_writer_obj, stop_event, adsb_reader=None, geojson_streamer=None, alert_manager=None):
    """Worker thread to process a single topic's frames."""
    print(f"Worker started for topic: {topic}")
    frames_count = 0
    try:
        while not stop_event.is_set():
            ok, frame = src.read()
            if not ok:
                print(f"Worker for {topic}: source exhausted or closed")
                break
            if frame is None:
                continue

            results = yolo.infer(frame)
            out = yolo.draw(frame, results)
            frames_count += 1
            
            # Get the actual frame timestamp from the source (for MCAP, this is the message log_time)
            frame_time_ns = getattr(src, 'last_log_time_ns', None)
            if frame_time_ns is None:
                frame_time_ns = int(time.time() * 1e9)  # Fallback to current time

            # Build protobuf
            state = build_plane_state(
                frame=frame,
                results=results,
                class_filter=args.class_filter,
                source_topic=topic,
                source_file=args.mcap or "",
            )

            # Emit protobuf detections
            if pb_writer:
                pb_writer.write(state)
            if plane_writer:
                mcap_writer_obj_local, channel_id = plane_writer
                t = time.time_ns()
                mcap_writer_obj_local.add_message(
                    channel_id=channel_id,
                    log_time=t,
                    publish_time=t,
                    data=state.SerializeToString(),
                )

            # Emit annotated video to MCAP
            if topic in video_writers:
                video_writers[topic].write(out)
            
            # Emit annotated video to Foxglove
            if topic in foxglove_video_writers:
                foxglove_video_writers[topic].write(out)
            
            # Emit confidence to Foxglove
            if topic in foxglove_confidence_writers:
                # Extract confidence from PlaneState
                confidence = state.plane.confidence if state.plane_present else None
                foxglove_confidence_writers[topic].write(confidence, label="airplane")
            
            # Stream ADS-B aircraft positions as GeoJSON
            # ADS-B and alert processing
            aircraft_states = {}
            if adsb_reader and geojson_streamer:
                try:
                    # Get aircraft at current frame timestamp (with lookback tolerance)
                    aircraft_states = adsb_reader.get_aircraft_at_time(frame_time_ns, lookback_secs=60)
                    if aircraft_states:
                        frame_time_s = frame_time_ns / 1e9
                        geojson_streamer.stream_aircraft(aircraft_states)
                    else:
                        if frames_count % 100 == 0:
                            frame_time_s = frame_time_ns / 1e9
                            logging.info(f"[{frames_count}] Frame time: {frame_time_s:.2f}s - No aircraft in view")
                except Exception as e:
                    logging.error(f"Failed to stream ADS-B data: {e}")
            
            # Update alert state
            if alert_manager:
                try:
                    # Collect YOLO detections from current frame
                    yolo_detections = []
                    if results and len(results) > 0:
                        for box in results[0].boxes:
                            conf = float(box.conf[0])
                            yolo_detections.append((topic, conf))
                    
                    # Update alert state with both ADS-B and YOLO data
                    frame_time_s = frame_time_ns / 1e9
                    alert_manager.update_alert_state(aircraft_states, yolo_detections, frame_time_s)
                except Exception as e:
                    logging.error(f"Failed to update alert state: {e}")

            # Display if single thread or forced
            if not args.no_display and len([src for src in [src]]) == 1:
                if args.show_fps and frames_count % 30 == 0:
                    print(f"{topic}: {frames_count} frames processed")
                cv2.imshow(f"YOLO {topic}", out)
                if cv2.waitKey(1) & 0xFF == 27:
                    stop_event.set()
                    break

    except Exception as e:
        print(f"Error in worker for {topic}: {e}")
        traceback.print_exc()
    finally:
        print(f"Worker finished for {topic}: processed {frames_count} frames")


if __name__ == "__main__":
    main()
