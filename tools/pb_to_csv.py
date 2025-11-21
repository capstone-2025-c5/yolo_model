# easy way to only print positives

import struct, sys, csv, pathlib
from third_party.protos_py import plane_detection_pb2 as pd_pb2

def read_len_prefixed(path):
  with open(path, "rb") as f:
    while True:
      hdr = f.read(4)
      if not hdr:
        return
      n = int.from_bytes(hdr, "big")
      yield f.read(n)

def main(in_pb, out_csv="out/detections_from_pb.csv"):
  pathlib.Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
  with open(out_csv, "w", newline="") as g:
    w = csv.writer(g)
    w.writerow(["frame_seq","timestamp_unix_ns","plane_present",
                "x","y","width","height","confidence",
                "image_width","image_height","source_file","source_topic"])
    for raw in read_len_prefixed(in_pb):
      m = pd_pb2.PlaneState()
      m.ParseFromString(raw)
      if m.plane_present:
        w.writerow([m.frame_seq, m.timestamp_unix_ns, True,
                    m.plane.x, m.plane.y, m.plane.width, m.plane.height, m.plane.confidence,
                    m.image_width, m.image_height, m.source_file, m.source_topic])
  print(f"Wrote {out_csv}")

if __name__ == "__main__":
  main(sys.argv[1] if len(sys.argv)>1 else "out/plane_states.pb")
