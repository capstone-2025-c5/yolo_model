# frame_source.py
# Pluggable frame sources: GStreamer window, OpenCV cam, DXCam window, Foxglove CompressedVideo (MCAP).
from typing import Optional
import ctypes
import ctypes.wintypes as wt
import cv2

# -----------------------------
# WinAPI helpers (ctypes only)
# -----------------------------
def find_window_handle(title_contains: str) -> Optional[int]:
    user32 = ctypes.windll.user32
    EnumWindows = user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    IsWindowVisible = user32.IsWindowVisible
    GetWindowTextW = user32.GetWindowTextW
    GetWindowTextLengthW = user32.GetWindowTextLengthW

    title_sub = (title_contains or "").lower()
    result_hwnd: int = 0

    def _callback(hwnd, lParam):
        nonlocal result_hwnd
        if IsWindowVisible(hwnd):
            length = GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                GetWindowTextW(hwnd, buf, length + 1)
                if title_sub in buf.value.lower():
                    result_hwnd = int(hwnd)
                    return False
        return True

    cb = EnumWindowsProc(_callback)
    EnumWindows(cb, 0)
    return result_hwnd or None

def bring_window_to_front(hwnd: int) -> None:
    if not hwnd:
        return
    user32 = ctypes.windll.user32
    SW_RESTORE = 9
    user32.ShowWindow(wt.HWND(hwnd), SW_RESTORE)
    user32.SetForegroundWindow(wt.HWND(hwnd))

# -----------------------------
# Base class
# -----------------------------
class FrameSource:
    def open(self) -> bool:
        raise NotImplementedError
    def read(self):
        raise NotImplementedError
    def release(self):
        pass

# -----------------------------
# GStreamer: Window-handle source (Option B)
# -----------------------------
class GStreamerWindowFrameSource(FrameSource):
    def __init__(self, title_contains: str, fps: int = 30):
        self.title_contains = title_contains
        self.fps = fps
        self.cap = None

    def build_pipeline(self, hwnd: int) -> str:
        hwnd_hex = hex(hwnd)
        return (
            f"d3d11screencapturesrc window-handle={hwnd_hex} window-capture-mode=client show-cursor=false ! "
            f"video/x-raw,framerate={self.fps}/1 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )

    def open(self) -> bool:
        hwnd = find_window_handle(self.title_contains)
        if not hwnd:
            raise RuntimeError(f"Could not find visible window containing: {self.title_contains!r}")
        try:
            bring_window_to_front(hwnd)
        except Exception:
            pass
        self.cap = cv2.VideoCapture(self.build_pipeline(hwnd), cv2.CAP_GSTREAMER)
        return self.cap.isOpened()

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

# -----------------------------
# OpenCV Camera source (future real camera)
# -----------------------------
class OpenCVCameraFrameSource(FrameSource):
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        return self.cap.isOpened()

    def read(self):
        if not self.cap:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap:
            self.cap.release()
            self.cap = None

# -----------------------------
# DXCam: Window-handle source (no GStreamer)
# -----------------------------
import numpy as np
import dxcam

class DXCamWindowFrameSource(FrameSource):
    def __init__(self, title_contains: str, fps: int = 30):
        self.title_contains = title_contains
        self.fps = fps
        self.camera = None
        self.hwnd = None
        self.region = None

    def open(self) -> bool:
        self.hwnd = find_window_handle(self.title_contains)
        if not self.hwnd:
            raise RuntimeError(f"Could not find visible window containing: {self.title_contains!r}")
        bring_window_to_front(self.hwnd)

        user32 = ctypes.windll.user32
        rect = wt.RECT()
        user32.GetClientRect(wt.HWND(self.hwnd), ctypes.byref(rect))
        pt = wt.POINT(0, 0)
        user32.ClientToScreen(wt.HWND(self.hwnd), ctypes.byref(pt))
        left, top = pt.x, pt.y
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        self.region = (left, top, left + width, top + height)

        self.camera = dxcam.create(output_idx=0, max_buffer_len=1)
        self.camera.start(region=self.region, target_fps=self.fps)
        return True

    def read(self):
        if not self.camera:
            return False, None
        frame = self.camera.get_latest_frame()
        if frame is None:
            return False, None
        if frame.ndim == 3:
            if frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return True, frame.copy(order="C")

    def release(self):
        if self.camera:
            self.camera.stop()
            self.camera = None

# -----------------------------
# MCAP foxglove.CompressedVideo (protobuf) with seek + H264/H265/JPEG decode
# -----------------------------
import time
import av  # PyAV
from mcap.reader import make_reader
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory as _mf

class MCAPFoxgloveCompressedVideoSource(FrameSource):
    """
    Reads an MCAP topic with schema 'foxglove.CompressedVideo' (protobuf),
    deserializes each message, and decodes frames:
      - format: 'h264'/'h265'/'hevc' -> PyAV (FFmpeg), AVCC→Annex-B normalized
      - format: 'jpeg'               -> cv2.imdecode

    Supports:
      - start_offset_s (float): seconds from start of recording
      - start_abs_ns (int): absolute log_time (ns since epoch)
      - preroll_s (float): back up before start to catch SPS/PPS/keyframes
      - realtime (bool): pace display by message timestamps if True
    """
    def __init__(
        self,
        mcap_path: str,
        topic: Optional[str] = None,
        realtime: bool = True,
        start_offset_s: Optional[float] = None,
        start_abs_ns: Optional[int] = None,
        preroll_s: float = 0.5,
    ):
        self.mcap_path = mcap_path
        self.topic = topic
        self.realtime = realtime
        self.start_offset_s = start_offset_s
        self.start_abs_ns = start_abs_ns
        self.preroll_s = max(0.0, float(preroll_s))

        self._buf = None
        self._reader = None
        self._iter = None
        self._prev_ns = None
        self._play_start_ns = None

        self._Msg = None            # protobuf message class
        self._decoder = None        # PyAV decoder for h264/hevc
        self._codec_name = None     # 'h264', 'hevc', or 'jpeg'

        self.recording_start_ns = None
        self.last_log_time_ns = None

    def _build_msg_class(self, schema) -> None:
        """
        Build dynamic protobuf message class for schema.name using FileDescriptorSet
        embedded in schema.data. Works with protobuf v3/v4 (MessageFactory.GetPrototype)
        and v5+ (message_factory.GetMessageClass).
        """
        fds = descriptor_pb2.FileDescriptorSet()
        fds.ParseFromString(schema.data)

        pool = descriptor_pool.DescriptorPool()
        for fd in fds.file:
            pool.Add(fd)

        desc = pool.FindMessageTypeByName(schema.name)  # "foxglove.CompressedVideo"
        if hasattr(_mf, "GetMessageClass"):
            # protobuf >= 5
            self._Msg = _mf.GetMessageClass(desc)  # type: ignore[attr-defined]
        else:
            # protobuf <= 4
            factory = _mf.MessageFactory(pool)
            self._Msg = factory.GetPrototype(desc)

    def _sniff_codec(self, fmt: str, buf: bytes) -> str:
        """Return 'jpeg', 'h264', or 'hevc' based on format string and magic bytes."""
        f = (fmt or "").lower()
        if "jpeg" in f or "jpg" in f or f.startswith("image/jpeg"):
            return "jpeg"
        if "265" in f or "hevc" in f or "hvc1" in f:
            return "hevc"
        if "264" in f or "avc" in f or "avc1" in f or "video/avc" in f:
            return "h264"
        # Byte sniff as fallback
        if buf.startswith(b"\xff\xd8\xff"):  # JPEG SOI
            return "jpeg"
        if b"\x00\x00\x00\x01" in buf or buf.startswith(b"\x00\x00\x01"):  # Annex-B start codes
            # Could be h264 or hevc; choose h264 (decoder will still handle if it's hevc and we reinit later)
            return "h264"
        return "h264"  # default best guess

    def _reinit_decoder(self, codec: str) -> None:
        """Ensure decoder matches requested codec ('h264','hevc','jpeg')."""
        if codec == self._codec_name:
            return
        # flush old decoder
        if self._decoder:
            try:
                for _ in self._decoder.decode(None):
                    pass
            except Exception:
                pass
        self._decoder = None
        if codec in ("h264", "hevc"):
            name = "h264" if codec == "h264" else "hevc"
            self._decoder = av.codec.CodecContext.create(name, "r")
        self._codec_name = codec

    @staticmethod
    def _avcc_to_annexb(avcc: bytes) -> bytes:
        """
        Convert length-prefixed NAL units (AVCC) to Annex-B start codes.
        Tries 4-, then 2-, then 1-byte NAL length fields.
        Returns original on malformed input or if already Annex-B.
        """
        # Already Annex-B?
        if avcc.startswith(b"\x00\x00\x00\x01") or avcc.startswith(b"\x00\x00\x01"):
            return avcc

        def convert(data: bytes, nlen: int) -> bytes | None:
            out = bytearray()
            i, n = 0, len(data)
            try:
                while i + nlen <= n:
                    ln = int.from_bytes(data[i:i+nlen], "big")
                    i += nlen
                    if ln <= 0 or i + ln > n:
                        return None  # malformed for this nlen
                    out += b"\x00\x00\x00\x01"
                    out += data[i:i+ln]
                    i += ln
                if i != n:
                    return None  # trailing bytes left over
                return bytes(out)
            except Exception:
                return None

        for nlen in (4, 2, 1):
            out = convert(avcc, nlen)
            if out is not None:
                return out

        # Give decoder a chance with original if we couldn't convert
        return avcc


    def open(self) -> bool:
        self._buf = open(self.mcap_path, "rb")
        self._reader = make_reader(self._buf)

        summary = self._reader.get_summary()
        if not summary.channels:
            raise RuntimeError("No channels in MCAP.")

        # Select channel (explicit topic > first foxglove.CompressedVideo > first channel)
        selected = None
        for ch in summary.channels.values():
            if self.topic and ch.topic == self.topic:
                selected = ch
                break
        if selected is None:
            for ch in summary.channels.values():
                sch = summary.schemas.get(ch.schema_id)
                if sch and sch.name == "foxglove.CompressedVideo":
                    selected = ch
                    break
        if selected is None:
            selected = next(iter(summary.channels.values()))

        schema = summary.schemas.get(selected.schema_id)
        if not schema or schema.name != "foxglove.CompressedVideo" or (schema.encoding or "").lower() != "protobuf":
            raise RuntimeError(
                f"Selected topic '{selected.topic}' doesn't have protobuf foxglove.CompressedVideo; "
                f"schema name={getattr(schema,'name',None)} encoding={getattr(schema,'encoding',None)}"
            )

        self._build_msg_class(schema)

        # Compute start + preroll
        stats = summary.statistics
        rec_start_ns = getattr(stats, "message_start_time", None)
        self.recording_start_ns = rec_start_ns
        start_ns = None
        if self.start_abs_ns is not None:
            start_ns = int(self.start_abs_ns)
        elif self.start_offset_s is not None and rec_start_ns is not None:
            start_ns = int(rec_start_ns + self.start_offset_s * 1e9)

        iter_start_ns = None
        if start_ns is not None:
            iter_start_ns = max(0, start_ns - int(self.preroll_s * 1e9))
            self._play_start_ns = start_ns

        self._iter = self._reader.iter_messages(
            topics=[selected.topic],
            start_time=iter_start_ns,
        )
        self._prev_ns = None
        self._decoder = None
        self._codec_name = None
        return True

    def read(self):
        if self._iter is None:
            return False, None
        try:
            _schema, _channel, msg = next(self._iter)
        except StopIteration:
            return False, None

        # Parse protobuf message
        m = self._Msg()
        m.ParseFromString(msg.data)
        self.last_log_time_ns = msg.log_time
        fmt = getattr(m, "format", "") or ""
        payload = bytes(getattr(m, "data", b"") or b"")

        # Preroll: feed decoder to gather SPS/PPS but don't emit frames
        if self._play_start_ns is not None and msg.log_time < self._play_start_ns:
            codec = self._sniff_codec(fmt, payload)
            self._reinit_decoder(codec)
            if self._decoder is not None and self._codec_name in ("h264", "hevc"):
                pkt = av.packet.Packet(self._avcc_to_annexb(payload))
                try:
                    _ = self._decoder.decode(pkt)
                except Exception:
                    pass
            return True, None

        # Realtime pacing
        if self.realtime:
            t_ns = msg.log_time
            if self._prev_ns is None:
                self._prev_ns = t_ns
            else:
                dt = (t_ns - self._prev_ns) / 1e9
                if dt > 0:
                    time.sleep(min(dt, 0.1))
                self._prev_ns = t_ns

        # Decode current message
        codec = self._sniff_codec(fmt, payload)
        self._reinit_decoder(codec)

        if self._codec_name == "jpeg":
            arr = np.frombuffer(payload, dtype=np.uint8)
            bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return True, bgr

        # H.264 / HEVC path: normalize to Annex-B then decode
        pkt = av.packet.Packet(self._avcc_to_annexb(payload))
        try:
            frames = self._decoder.decode(pkt)
        except Exception:
            # last-resort retry with raw payload
            try:
                frames = self._decoder.decode(av.packet.Packet(payload))
            except Exception:
                return True, None
        if not frames:
            return True, None
        frame = frames[-1]
        return True, frame.to_ndarray(format="bgr24")

    def release(self):
        # Flush decoder
        if self._decoder:
            try:
                for _ in self._decoder.decode(None):
                    pass
            except Exception:
                pass
        if self._buf:
            try:
                self._buf.close()
            except Exception:
                pass
        self._decoder = None
        self._reader = None
        self._iter = None
        self._buf = None
        self._prev_ns = None
        self._play_start_ns = None
