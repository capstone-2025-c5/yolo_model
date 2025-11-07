import time
import argparse
import cv2
import config as C
from frame_source import (
    GStreamerWindowFrameSource,
    OpenCVCameraFrameSource,
    DXCamWindowFrameSource,
    MCAPFoxgloveCompressedVideoSource,
)
from yolo_infer import YOLORunner

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

def main():
    ap = argparse.ArgumentParser(description="Live YOLO inference from pluggable sources")
    ap.add_argument("--source", default=C.DEFAULT_SOURCE, choices=["gst_window", "dx_window", "camera", "mcap"])
    ap.add_argument("--camera-index", type=int, default=0)

    # MCAP args
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
    ap.add_argument("--class-filter", nargs="*", default=None)
    args = ap.parse_args()

    src = make_source(args)
    print("Source type:", type(src).__name__)
    if not src.open():
        raise RuntimeError("Failed to open frame source")

    yolo = YOLORunner(weights=args.weights, conf=args.conf, imgsz=args.imgsz, allowed_names=args.class_filter)

    prev, frames, fps = time.time(), 0, 0.0
    while True:
        ok, frame = src.read()
        if not ok:
            break
        if frame is None:
            continue

        results = yolo.infer(frame)
        out = yolo.draw(frame, results)

        frames += 1
        now = time.time()
        if now - prev >= 1.0:
            fps = frames / (now - prev)
            frames, prev = 0, now
        if args.show_fps:
            cv2.putText(out, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
        cv2.imshow("YOLO Live", out)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    src.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
