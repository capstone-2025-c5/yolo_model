import time
import cv2
import argparse
from frame_source import GStreamerWindowFrameSource, OpenCVCameraFrameSource, DXCamWindowFrameSource
import config as C

def make_source(args):
    if args.source == "gst_window":
        return GStreamerWindowFrameSource(C.TARGET_TITLE_CONTAINS, fps=C.TARGET_FPS)
    elif args.source == "dx_window":
        return DXCamWindowFrameSource(C.TARGET_TITLE_CONTAINS, fps=C.TARGET_FPS)
    elif args.source == "camera":
        return OpenCVCameraFrameSource(index=args.camera_index, width=C.TARGET_W, height=C.TARGET_H, fps=C.TARGET_FPS)
    else:
        raise ValueError("Unknown source")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=C.DEFAULT_SOURCE, choices=["gst_window", "dx_window", "camera"])
    ap.add_argument("--camera-index", type=int, default=0)
    args = ap.parse_args()

    src = make_source(args)
    if not src.open():
        raise RuntimeError("Failed to open frame source")

    prev, frames = time.time(), 0
    fps = 0.0

    while True:
        ok, frame = src.read()
        if not ok:
            print("No frame; is the window visible or the camera connected?")
            break

        frames += 1
        now = time.time()
        if now - prev >= 1.0:
            fps = frames / (now - prev)
            frames = 0
            prev = now
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

        cv2.imshow("Capture Preview", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    src.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
