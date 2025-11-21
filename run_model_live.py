# run_model_live.py
# Capture → YOLO → display (+ Protobuf/MCAP emission)

import time
import argparse
import cv2
import os
from typing import Iterable, Tuple, Optional

import config as C
from frame_source import (
    GStreamerWindowFrameSource,
    OpenCVCameraFrameSource,
    DXCamWindowFrameSource,
    MCAPFoxgloveCompressedVideoSource,
)
from yolo_infer import YOLORunner
from third_party.protos_py import plane_detection_pb2 as pd_pb2

# --- Optional MCAP support (only if --out-mcap is provided) ---
_MCAP_AVAILABLE = True
try:
    from mcap.mcap0.writer import Writer as McapWriter  # type: ignore
    from google.protobuf import descriptor_pb2  # for schema FileDescriptorSet
except Exception:
    _MCAP_AVAILABLE = False


def make_source(args):
    if args.source == "gst_window":
        return GStreamerWindowFrameSource(C.TARGET_TITLE_CONTAINS, fps=C.TARGET_FPS)
    elif args.source == "dx_window":
        return DXCamWindowFrameSource(C.TARGET_TITLE_CONTAINS, fps=C.TARGET_FPS)
    elif args.source == "camera":
        return OpenCVCameraFrameSource(index=args.camera_index, width=C.TARGET_W, height=C.TARGET_H, fps=C.TARGET_FPS)
    elif args.source == "mcap":
        if not args.mcap:
            raise ValueError("When --source mcap, you must provide --mcap <path to .mcap>")
        return MCAPFoxgloveCompressedVideoSource(
            mcap_path=args.mcap,
            topic=args.topic,
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


def main():
    ap = argparse.ArgumentParser(description="Live YOLO inference from pluggable sources + protobuf emission")
    ap.add_argument("--source", default=C.DEFAULT_SOURCE, choices=["gst_window", "dx_window", "camera", "mcap"])
    ap.add_argument("--camera-index", type=int, default=0)

    # MCAP input (video) args
    ap.add_argument("--mcap", default=None)
    ap.add_argument("--topic", default=None)
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
    ap.add_argument("--out-mcap", default=None, help="Write PlaneState messages to this MCAP file")
    ap.add_argument("--topic-out", default="/detect/plane_state", help="Output topic name for MCAP")

    # Display toggle
    ap.add_argument("--no-display", action="store_true", help="Disable display window for headless runs")

    args = ap.parse_args()

    # Ensure output dirs exist if provided
    if args.out_pb:
        d = os.path.dirname(args.out_pb)
        if d:
            os.makedirs(d, exist_ok=True)
    if args.out_mcap:
        d = os.path.dirname(args.out_mcap)
        if d:
            os.makedirs(d, exist_ok=True)

    # Frame source
    src = make_source(args)
    print("Source type:", type(src).__name__)
    if not src.open():
        raise RuntimeError("Failed to open frame source")

    # Model
    yolo = YOLORunner(weights=args.weights, conf=args.conf, imgsz=args.imgsz, allowed_names=args.class_filter)

    # Outputs
    pb_writer = PBStreamWriter(args.out_pb) if args.out_pb else None
    mcap_writer = MCAPPlaneWriter(args.out_mcap, args.topic_out, pd_pb2) if args.out_mcap else None

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

            # Emit protobuf(s)
            if pb_writer:
                pb_writer.write(state)
            if mcap_writer:
                mcap_writer.write(state)

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
        src.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        if pb_writer:
            pb_writer.close()
        if mcap_writer:
            mcap_writer.close()


if __name__ == "__main__":
    main()
