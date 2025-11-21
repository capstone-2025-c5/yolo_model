# run_mcap_batch_detect.py
# Batch-scan MCAP videos (foxglove.CompressedVideo) for "airplane" presence,
# printing enter/exit times (offsets). Runs headless and as fast as possible.

import argparse
import os
import glob
import time

import numpy as np
import cv2

from frame_source import MCAPFoxgloveCompressedVideoSource
from yolo_infer import YOLORunner   # uses your existing wrapper
import config as C


def parse_ultralytics_results(results, allowed_names, conf_thres: float) -> bool:
    """
    Return True if any detection in Ultralytics-style Results matches allowed_names @ conf_thres.
    Works with ultralytics>=8 typical outputs and tries to be defensive.
    """
    try:
        # Many wrappers return a list with one Results per image
        r = results[0] if isinstance(results, (list, tuple)) else results

        # Ultralytics: r.boxes with .cls (tensor), .conf (tensor), and r.names dict
        boxes = getattr(r, "boxes", None)
        names = getattr(r, "names", None)
        if boxes is not None and names is not None:
            cls = getattr(boxes, "cls", None)
            con = getattr(boxes, "conf", None)
            if cls is not None and con is not None:
                cls_idx = cls.detach().cpu().numpy().astype(int)
                confs = con.detach().cpu().numpy()
                for i in range(len(cls_idx)):
                    name = names.get(int(cls_idx[i]), "").lower()
                    if name in allowed_names and confs[i] >= conf_thres:
                        return True
                return False

        # Fallback: try iterable of dicts with "name"/"confidence"
        if isinstance(r, (list, tuple)):
            for det in r:
                name = str(det.get("name", "")).lower()
                conf = float(det.get("confidence", 0.0))
                if name in allowed_names and conf >= conf_thres:
                    return True
            return False

    except Exception:
        # Be robust—on unknown result shapes, assume no match rather than crash
        return False

    return False


def scan_file(
    mcap_path: str,
    topic: str,
    yolo: YOLORunner,
    allowed_names: set,
    conf: float,
    gap_sec: float,
    start_offset: float,
    preroll: float,
    stride: int,
) -> None:
    print(f"\n=== Scanning: {os.path.basename(mcap_path)} ===")

    src = MCAPFoxgloveCompressedVideoSource(
        mcap_path=mcap_path,
        topic=topic,
        realtime=False,                 # run as fast as possible
        start_offset_s=start_offset,
        start_abs_ns=None,
        preroll_s=preroll,
    )
    if not src.open():
        print(f"Failed to open {mcap_path}")
        return

    in_event = False
    last_hit_ns = None
    frames_seen = 0

    # Convenience to compute offsets (seconds from recording start)
    def offset_s(ns: int) -> float:
        if src.recording_start_ns is None or ns is None:
            return 0.0
        return max(0.0, (ns - src.recording_start_ns) / 1e9)

    try:
        while True:
            ok, frame = src.read()
            if not ok:
                break
            if frame is None:
                continue

            frames_seen += 1
            if stride > 1 and (frames_seen % stride) != 0:
                continue  # skip frames for speed

            # Run YOLO
            results = yolo.infer(frame)
            hit = parse_ultralytics_results(results, allowed_names, conf)

            t_ns = src.last_log_time_ns  # set by the frame_source read()
            if hit:
                last_hit_ns = t_ns
                if not in_event:
                    in_event = True
                    print(f"plane detected at offset: {offset_s(t_ns):.3f}s for file: {os.path.basename(mcap_path)}")
            else:
                if in_event and last_hit_ns is not None:
                    # if we've been without a hit for > gap_sec, close the event
                    if t_ns is not None and (t_ns - last_hit_ns) >= gap_sec * 1e9:
                        in_event = False
                        print(f"plane no longer detected at offset: {offset_s(t_ns):.3f}s for file: {os.path.basename(mcap_path)}")

        # At EOF, if still in_event, close it at the last time we saw a hit (or now if unknown)
        if in_event:
            t_close = last_hit_ns or src.last_log_time_ns
            print(f"plane no longer detected at offset: {offset_s(t_close):.3f}s for file: {os.path.basename(mcap_path)}")

    finally:
        src.release()


def main():
    ap = argparse.ArgumentParser(description="Batch MCAP airplane detection (headless, fast)")
    ap.add_argument("--mcap-dir", default=None, help="Directory with .mcap files (process all)")
    ap.add_argument("--mcap", default=None, help="Single .mcap file (if you don't use --mcap-dir)")
    ap.add_argument("--topic", required=True, help="foxglove.CompressedVideo topic (e.g., /camera/0/image/compressed)")
    ap.add_argument("--weights", default=C.YOLO_WEIGHTS, help="YOLO weights")
    ap.add_argument("--conf", type=float, default=C.YOLO_CONF, help="Confidence threshold")
    ap.add_argument("--imgsz", type=int, default=C.YOLO_IMGSZ, help="Inference image size")
    ap.add_argument("--gap-sec", type=float, default=3.5, help="Silence gap to end an event")
    ap.add_argument("--start-offset", type=float, default=0.0, help="Start offset (seconds into recording)")
    ap.add_argument("--preroll", type=float, default=1.0, help="Preroll seconds before start to catch SPS/PPS")
    ap.add_argument("--stride", type=int, default=1, help="Process every Nth frame for extra speed")
    ap.add_argument("--classes", nargs="*", default=["airplane"], help="Allowed class names (default: airplane)")
    args = ap.parse_args()

    # Init YOLO (headless)
    yolo = YOLORunner(weights=args.weights, conf=args.conf, imgsz=args.imgsz, allowed_names=args.classes)

    # Build allowed-names set directly from CLI (don’t rely on YOLORunner having an attribute)
    allowed = set(s.lower() for s in (args.classes or ["airplane"]))

    files = []
    if args.mcap_dir:
        files = sorted(glob.glob(os.path.join(args.mcap_dir, "*.mcap")))
    elif args.mcap:
        files = [args.mcap]
    else:
        print("Provide --mcap-dir or --mcap")
        return

    if not files:
        print("No .mcap files found.")
        return

    for f in files:
        scan_file(
            mcap_path=f,
            topic=args.topic,
            yolo=yolo,
            allowed_names=allowed,
            conf=args.conf,
            gap_sec=args.gap_sec,
            start_offset=args.start_offset,
            preroll=args.preroll,
            stride=max(1, args.stride),
        )


if __name__ == "__main__":
    main()
