# run_model_live.py
# Capture → YOLO → display
# Sources supported:
#   --source gst_window  (GStreamer window-handle; requires OpenCV+GStreamer + GS 1.24+)
#   --source dx_window   (DXCam window-handle; recommended on Windows)
#   --source camera      (OpenCV webcam)

import time
import argparse
import cv2

from frame_source import (
    GStreamerWindowFrameSource,
    OpenCVCameraFrameSource,
    DXCamWindowFrameSource,
)
from yolo_infer import YOLORunner
import config as C


def make_source(name: str, cam_idx: int):
    if name == "gst_window":
        return GStreamerWindowFrameSource(C.TARGET_TITLE_CONTAINS, fps=C.TARGET_FPS)
    elif name == "dx_window":
        return DXCamWindowFrameSource(C.TARGET_TITLE_CONTAINS, fps=C.TARGET_FPS)
    elif name == "camera":
        return OpenCVCameraFrameSource(
            index=cam_idx, width=C.TARGET_W, height=C.TARGET_H, fps=C.TARGET_FPS
        )
    else:
        raise ValueError(f"Unknown source: {name}")


def main():
    ap = argparse.ArgumentParser(description="Live YOLO inference from pluggable sources")
    ap.add_argument("--source", default=C.DEFAULT_SOURCE, choices=["gst_window", "dx_window", "camera"])
    ap.add_argument("--camera-index", type=int, default=0, help="Camera index for --source camera")
    ap.add_argument("--weights", default=C.YOLO_WEIGHTS, help="Path to YOLO weights (e.g., yolov8n.pt)")
    ap.add_argument("--conf", type=float, default=C.YOLO_CONF, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=C.YOLO_IMGSZ, help="Inference image size")
    ap.add_argument("--show-fps", dest="show_fps", action="store_true", help="Overlay FPS counter")
    ap.add_argument("--class-filter", nargs="*", default=None, help="Optional list of class names to keep (e.g. --class-filter airplane)")
    args = ap.parse_args()

    # Frame source
    src = make_source(args.source, args.camera_index)
    if not src.open():
        raise RuntimeError("Failed to open frame source")

    # Model
    yolo = YOLORunner(weights=args.weights, conf=args.conf, imgsz=args.imgsz, allowed_names=args.class_filter)

    prev, frames = time.time(), 0
    fps = 0.0

    # Optional filtering by class name(s)
    allowed = set(s.lower() for s in args.class_filter) if args.class_filter else None

    while True:
        ok, frame = src.read()
        if not ok or frame is None:
            print("No frame; is the window visible (not minimized) or the camera connected?")
            break

        results = yolo.infer(frame)
        r = results[0]

        out = yolo.draw(frame, results)

        frames += 1
        now = time.time()
        if now - prev >= 1.0:
            fps = frames / (now - prev)
            frames = 0
            prev = now

        # Use the correct underscore attribute
        if args.show_fps:
            cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        cv2.imshow("YOLO Live", out)
        if cv2.waitKey(1) & 0xFF == 27:  # Esc to quit
            break

    src.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
