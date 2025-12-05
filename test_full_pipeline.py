#!/usr/bin/env python3
"""Test full pipeline: YOLO + ADS-B GeoJSON streaming to Foxglove."""

import os
import sys

def main():
    adsb_path = os.path.expanduser("~/Downloads/adsb-decodes/20251025-103624-adsb-decode.mcap")
    video_path = os.path.expanduser("~/Downloads/merged_spliced.mcap")
    
    print("=" * 70)
    print("YOLO + ADS-B GeoJSON Streaming to Foxglove")
    print("=" * 70)
    print()
    print("Command:")
    print()
    
    cmd = f"""python ./run_model_live.py \\
  --source mcap \\
  --mcap {video_path} \\
  --topics /camera/0/image/compressed /camera/1/image/compressed \\
  --start-offset 145 \\
  --preroll 2.0 \\
  --no-realtime \\
  --class-filter airplane \\
  --conf 0.10 \\
  --foxglove-host 0.0.0.0 \\
  --foxglove-port 8765 \\
  --adsb-mcap {adsb_path} \\
  --adsb-geojson-topic /aircraft/positions \\
  --no-display"""
    
    print(cmd)
    print()
    print("This will stream:")
    print("  • Video detections with bounding boxes to:")
    print("    - /camera/0/image/compressed/detections")
    print("    - /camera/1/image/compressed/detections")
    print("  • YOLO confidence scores to:")
    print("    - /camera/0/image/compressed/confidence")
    print("    - /camera/1/image/compressed/confidence")
    print("  • ADS-B aircraft positions as GeoJSON to:")
    print("    - /aircraft/positions")
    print()
    print("Connect Foxglove viewer to ws://0.0.0.0:8765")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
