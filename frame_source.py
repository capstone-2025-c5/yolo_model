# frame_source.py
# Pluggable frame sources for live pipelines.
# - GStreamerWindowFrameSource: captures a specific window using GStreamer on Windows (Option B).
# - OpenCVCameraFrameSource: standard webcam via OpenCV (future real camera swap-in).
#
# This version removes the pywin32 dependency and uses pure WinAPI via ctypes.

from typing import Optional
import cv2
import ctypes
import ctypes.wintypes as wt

# -----------------------------
# WinAPI helpers (ctypes only)
# -----------------------------

def find_window_handle(title_contains: str) -> Optional[int]:
    """
    Find the first visible top-level window whose title contains the given substring.
    Returns the HWND as an int, or None if not found.
    """
    user32 = ctypes.windll.user32
    EnumWindows = user32.EnumWindows
    EnumWindowsProc = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    IsWindowVisible = user32.IsWindowVisible
    GetWindowTextW = user32.GetWindowTextW
    GetWindowTextLengthW = user32.GetWindowTextLengthW

    title_sub = (title_contains or "").lower()
    result_hwnd: int = 0  # keep as plain int

    def _callback(hwnd, lParam):
        nonlocal result_hwnd
        if IsWindowVisible(hwnd):
            length = GetWindowTextLengthW(hwnd)
            if length > 0:
                buf = ctypes.create_unicode_buffer(length + 1)
                GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value
                if title_sub in title.lower():
                    result_hwnd = int(hwnd)
                    return False  # stop enumeration
        return True  # continue

    cb = EnumWindowsProc(_callback)
    EnumWindows(cb, 0)
    return result_hwnd or None


def bring_window_to_front(hwnd: int) -> None:
    """Best-effort: restore and bring a window to the foreground."""
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
    """
    Captures frames from a specific window using the GStreamer element:
      d3d11screencapturesrc window-handle=... window-capture-mode=client
    Produces BGR frames via appsink for OpenCV.

    Requirements:
      - Windows 10/11
      - GStreamer 1.24+ (for window-handle on d3d11screencapturesrc)
      - GStreamer bin dir on PATH
      - OpenCV built with GStreamer support (opencv-python wheels typically OK)
    """
    def __init__(self, title_contains: str, fps: int = 30):
        self.title_contains = title_contains
        self.fps = fps
        self.cap = None

    def build_pipeline(self, hwnd: int) -> str:
        hwnd_hex = hex(hwnd)
        # Capture only the client area; hide cursor; convert to BGR for OpenCV; deliver via appsink.
        pipeline = (
            f"d3d11screencapturesrc window-handle={hwnd_hex} window-capture-mode=client show-cursor=false ! "
            f"video/x-raw,framerate={self.fps}/1 ! "
            f"videoconvert ! video/x-raw,format=BGR ! "
            f"appsink drop=true max-buffers=1 sync=false"
        )
        return pipeline

    def open(self) -> bool:
        hwnd = find_window_handle(self.title_contains)
        if not hwnd:
            raise RuntimeError(
                f"Could not find a visible window with title containing: {self.title_contains!r}. "
                "Ensure the app is running in windowed mode and not minimized."
            )

        # Optional: bring it to front (helps some capture paths)
        try:
            bring_window_to_front(hwnd)
        except Exception:
            pass

        pipeline = self.build_pipeline(hwnd)
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        ok = self.cap.isOpened()

        return ok

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
    """
    Standard webcam capture using OpenCV. Good stand-in for future real camera input.
    """
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30):
        self.index = index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def open(self) -> bool:
        # On Windows, CAP_DSHOW is often reliable; CAP_MSMF is another option.
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW)
        # Try to set properties (driver may ignore these)
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
# DXCam: Window-handle source (no GStreamer needed)
# -----------------------------
import numpy as np
import dxcam

class DXCamWindowFrameSource(FrameSource):
    """
    Captures frames from a specific window using Windows Desktop Duplication (dxcam).
    Produces BGR frames (numpy arrays) compatible with OpenCV.
    """
    def __init__(self, title_contains: str, fps: int = 30):
        self.title_contains = title_contains
        self.fps = fps
        self.camera = None
        self.hwnd = None
        self.region = None

    def open(self) -> bool:
        self.hwnd = find_window_handle(self.title_contains)
        if not self.hwnd:
            raise RuntimeError(
                f"Could not find a visible window with title containing: {self.title_contains!r}. "
                "Ensure the app is windowed and not minimized."
            )
        bring_window_to_front(self.hwnd)

        # Build client-area screen coordinates for this window
        user32 = ctypes.windll.user32
        rect = wt.RECT()
        user32.GetClientRect(wt.HWND(self.hwnd), ctypes.byref(rect))
        pt = wt.POINT(0, 0)
        user32.ClientToScreen(wt.HWND(self.hwnd), ctypes.byref(pt))
        left, top = pt.x, pt.y
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        self.region = (left, top, left + width, top + height)

        # Create and start dxcam
        self.camera = dxcam.create(output_idx=0, max_buffer_len=1)
        self.camera.start(region=self.region, target_fps=self.fps)
        return True

    def read(self):
        if not self.camera:
            return False, None
        frame = self.camera.get_latest_frame()
        if frame is None:
            return False, None

        # Normalize to BGR for OpenCV drawing
        if frame.ndim == 3:
            if frame.shape[-1] == 4:
                # BGRA (most common) or RGBA depending on system
                # Try BGRA->BGR first; if colors still look off, switch to RGBA->BGR
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif frame.shape[-1] == 3:
                # Likely RGB -> convert to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Make sure memory is contiguous for cv2 ops
        frame = frame.copy(order="C")
        return True, frame

    def release(self):
        if self.camera:
            self.camera.stop()
            self.camera = None
