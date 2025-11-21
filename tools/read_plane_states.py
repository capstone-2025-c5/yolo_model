# tools/read_plane_states.py
import struct, sys, pathlib
from third_party.protos_py import plane_detection_pb2 as pd_pb2

def read_len_prefixed(f):
    while True:
        hdr = f.read(4)
        if not hdr:
            return
        if len(hdr) < 4:
            raise EOFError("Truncated length header")
        (n,) = struct.unpack(">I", hdr)  # uint32 big-endian
        body = f.read(n)
        if len(body) != n:
            raise EOFError("Truncated message body")
        yield body

def main(path):
    p = pathlib.Path(path)
    if not p.exists():
        print(f"File not found: {p}")
        return

    total = 0
    present = 0
    first_n = 10
    with p.open("rb") as f:
        for i, raw in enumerate(read_len_prefixed(f), start=1):
            msg = pd_pb2.PlaneState()
            msg.ParseFromString(raw)
            total += 1
            if msg.plane_present:
                present += 1
                if i <= first_n:
                    print(
                        f"[{i}] present=True  x={msg.plane.x:.3f} y={msg.plane.y:.3f} "
                        f"w={msg.plane.width:.3f} h={msg.plane.height:.3f} conf={msg.plane.confidence:.2f}"
                    )
            else:
                if i <= first_n:
                    print(f"[{i}] present=False")
    print(f"\nSummary: {total} messages, {present} with plane_present=True")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "out/plane_states.pb"
    main(path)

