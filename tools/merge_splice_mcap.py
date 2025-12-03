# tools/merge_splice_mcap.py
# Merge-splice multiple MCAPs into a single MCAP, keeping original topic names.
# Adds: keyframe-safe splicing for H.264/H.265 and option to preserve original times.

import argparse
import csv
import pathlib
from typing import Dict, List, Tuple, Optional

# MCAP imports (support multiple package layouts)
try:
    from mcap.writer import Writer as McapWriter  # mcap >= 1.2
except ImportError:
    from mcap.mcap0.writer import Writer as McapWriter  # older mcap
from mcap.reader import make_reader

# --------- tiny H26x helpers (Annex B start-code scanning) ----------
_START3 = b"\x00\x00\x01"
_START4 = b"\x00\x00\x00\x01"

def _split_nalus(b: bytes):
    """Yield (offset, nalu_bytes) for AnnexB stream."""
    i = 0
    n = len(b)
    # find first start code
    while i < n - 3 and b[i:i+3] != _START3 and b[i:i+4] != _START4:
        i += 1
    while i < n:
        # determine start length
        if b[i:i+4] == _START4:
            sc_len = 4
        elif b[i:i+3] == _START3:
            sc_len = 3
        else:
            break
        j = i + sc_len
        # find next start code
        k = j
        while k < n - 3 and b[k:k+3] != _START3 and b[k:k+4] != _START4:
            k += 1
        yield j, b[j:k]
        i = k

def _h264_type(nalu: bytes) -> int:
    return nalu[0] & 0x1F if nalu else -1  # 7=SPS, 8=PPS, 5=IDR

def _h265_type(nalu: bytes) -> int:
    # first two bytes: f(1) + type(6) + layer(6) + tid(3)
    return ((nalu[0] >> 1) & 0x3F) if len(nalu) >= 2 else -1
# h265: 32 VPS, 33 SPS, 34 PPS, 19/20 IDR types

def _is_h264_idr(data: bytes) -> bool:
    for _, nalu in _split_nalus(data):
        t = _h264_type(nalu)
        if t == 5:
            return True
    return False

def _is_h265_idr(data: bytes) -> bool:
    for _, nalu in _split_nalus(data):
        t = _h265_type(nalu)
        if t in (19, 20):
            return True
    return False

def _collect_sps_pps_h264(data: bytes):
    sps, pps = [], []
    for _, nalu in _split_nalus(data):
        t = _h264_type(nalu)
        if t == 7:
            sps.append(nalu)
        elif t == 8:
            pps.append(nalu)
    # return as AnnexB blob
    if not (sps or pps):
        return None
    out = bytearray()
    for n in sps:
        out += _START4 + n
    for n in pps:
        out += _START4 + n
    return bytes(out)

def _collect_vps_sps_pps_h265(data: bytes):
    vps, sps, pps = [], [], []
    for _, nalu in _split_nalus(data):
        t = _h265_type(nalu)
        if t == 32:
            vps.append(nalu)
        elif t == 33:
            sps.append(nalu)
        elif t == 34:
            pps.append(nalu)
    if not (vps or sps or pps):
        return None
    out = bytearray()
    for n in vps:
        out += _START4 + n
    for n in sps:
        out += _START4 + n
    for n in pps:
        out += _START4 + n
    return bytes(out)
# --------------------------------------------------------------------

def _parse_topics(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [t.strip() for t in s.split(",") if t.strip()]

def _load_windows(csv_path: str) -> Dict[str, List[Tuple[float, float]]]:
    by_file: Dict[str, List[Tuple[float, float]]] = {}
    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        headers = {h.lower(): h for h in (r.fieldnames or [])}
        k_file = headers.get("file") or headers.get("filename")
        k_start = headers.get("start_offset_s") or headers.get("start") or headers.get("start_s")
        k_end = headers.get("end_offset_s") or headers.get("end") or headers.get("end_s")
        if not (k_file and k_start and k_end):
            raise ValueError(f"CSV must have file/filename, start_offset_s, end_offset_s (got: {r.fieldnames})")
        for row in r:
            name = row[k_file].strip()
            if not name:
                continue
            try:
                s = float(row[k_start]); e = float(row[k_end])
            except Exception:
                continue
            if e <= s:
                continue
            by_file.setdefault(name, []).append((s, e))
    # merge overlaps per file
    for fn, ivals in by_file.items():
        ivals.sort()
        merged: List[Tuple[float, float]] = []
        for s, e in ivals:
            if not merged or s > merged[-1][1]:
                merged.append((s, e))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        by_file[fn] = merged
    return by_file

def _merge_overlaps_ns(intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    out = [intervals[0]]
    for s, e in intervals[1:]:
        if s > out[-1][1]:
            out.append((s, e))
        else:
            out[-1] = (out[-1][0], max(out[-1][1], e))
    return out

def _within_any(t: int, intervals: List[Tuple[int, int]]) -> bool:
    for s, e in intervals:
        if t < s:
            return False
        if s <= t <= e:
            return True
    return False

def main():
    ap = argparse.ArgumentParser(description="Merge-splice MCAPs into one file, keeping original topics")
    ap.add_argument("--mcap-dir", required=True)
    ap.add_argument("--csv", required=True, help="Detections CSV (per-file start/end offsets)")
    ap.add_argument("--topics", default=None,
                    help="Comma-separated topics to keep (omit to keep all).")
    ap.add_argument("--out", default="out/spliced/merged_spliced.mcap")
    ap.add_argument("--preroll", type=float, default=2.0, help="Seconds before each window start")
    ap.add_argument("--postroll", type=float, default=0.0, help="Seconds after each window end")
    ap.add_argument("--preserve-times", action="store_true",
                    help="Keep original log/publish times (recommended for fidelity)")
    ap.add_argument("--start-on-keyframe", action="store_true", default=True,
                    help="Wait for SPS/PPS + IDR before starting each kept segment")
    ap.add_argument("--gap-ns", type=int, default=1_000_000,  # 1 ms
                    help="Gap between concatenated clips when not preserving times (ns)")
    args = ap.parse_args()

    mcap_dir = pathlib.Path(args.mcap_dir)
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    topics = _parse_topics(args.topics)

    windows_by_file = _load_windows(args.csv)
    name_to_path: Dict[str, pathlib.Path] = {p.name: p for p in mcap_dir.glob("*.mcap")}

    missing = [n for n in windows_by_file.keys() if n not in name_to_path]
    if missing:
        print("[WARN] CSV files not found in --mcap-dir:")
        for n in missing:
            print("   ", n)

    with open(out_path, "wb") as f_out:
        writer = McapWriter(f_out)
        writer.start(profile="", library="merge_splice_mcap 2.0")

        try:
            schema_key_to_id: Dict[Tuple[str, str, bytes], int] = {}
            topic_key_to_id: Dict[Tuple[str, int, str], int] = {}
            concat_base_ns = 0
            gap_ns = max(0, int(args.gap_ns))

            for csv_name, ivals_s in windows_by_file.items():
                in_path = name_to_path.get(csv_name)
                if not in_path:
                    continue

                with open(in_path, "rb") as f_in:
                    reader = make_reader(f_in)
                    summary = reader.get_summary()
                    if not summary.channels:
                        print(f"[WARN] No channels in {in_path}")
                        continue

                    stats = summary.statistics
                    rec_start_ns = getattr(stats, "message_start_time", None)
                    if rec_start_ns is None:
                        it = reader.iter_messages()
                        try:
                            _, _, first_msg = next(it)
                            rec_start_ns = first_msg.log_time
                        except StopIteration:
                            rec_start_ns = 0

                    pre_ns = int(max(0.0, args.preroll) * 1e9)
                    post_ns = int(max(0.0, args.postroll) * 1e9)
                    ivals_ns: List[Tuple[int, int]] = []
                    for s, e in ivals_s:
                        s_ns = int(rec_start_ns + s * 1e9) - pre_ns
                        e_ns = int(rec_start_ns + e * 1e9) + post_ns
                        if e_ns > s_ns:
                            ivals_ns.append((max(0, s_ns), e_ns))
                    ivals_ns = _merge_overlaps_ns(ivals_ns)

                    def ensure_schema(src_schema_id: Optional[int]) -> int:
                        if src_schema_id is None:
                            return 0
                        schema = summary.schemas.get(src_schema_id)
                        if schema is None:
                            return 0
                        key = (schema.name, schema.encoding or "", schema.data or b"")
                        if key in schema_key_to_id:
                            return schema_key_to_id[key]
                        new_id = writer.register_schema(
                            name=schema.name,
                            encoding=schema.encoding,
                            data=schema.data,
                        )
                        schema_key_to_id[key] = new_id
                        return new_id

                    def ensure_channel(src_chan_id: int) -> int:
                        ch = summary.channels.get(src_chan_id)
                        if ch is None:
                            key = (f"/unknown/{src_chan_id}", 0, "")
                            if key in topic_key_to_id:
                                return topic_key_to_id[key]
                            new_id = writer.register_channel(topic=key[0], message_encoding=key[2], schema_id=key[1])
                            topic_key_to_id[key] = new_id
                            return new_id
                        if topics is not None and ch.topic not in topics:
                            pass
                        new_schema_id = ensure_schema(ch.schema_id)
                        key = (ch.topic, new_schema_id, ch.message_encoding or "")
                        if key in topic_key_to_id:
                            return topic_key_to_id[key]
                        new_id = writer.register_channel(
                            topic=ch.topic,
                            message_encoding=ch.message_encoding,
                            schema_id=new_schema_id,
                        )
                        topic_key_to_id[key] = new_id
                        return new_id

                    # Track codec state per topic to start cleanly on each interval
                    last_codec: Dict[str, str] = {}          # topic -> "h264"/"hevc"/"jpeg"/"other"
                    last_ppsblob: Dict[str, bytes] = {}      # topic -> SPS/PPS (AnnexB), or VPS+SPS+PPS for h265
                    gating_on_key: Dict[str, bool] = {}      # topic -> currently waiting for IDR?

                    msg_iter = reader.iter_messages(topics=topics) if topics else reader.iter_messages()
                    current_interval_start_ns = None

                    for schema, channel, message in msg_iter:
                        t = message.log_time
                        if not _within_any(t, ivals_ns):
                            if not args.preserve_times and current_interval_start_ns is not None:
                                concat_base_ns += gap_ns
                                current_interval_start_ns = None
                                gating_on_key.clear()
                            continue

                        # entering new kept interval?
                        if not args.preserve_times and current_interval_start_ns is None:
                            current_interval_start_ns = t
                            gating_on_key.clear()  # new interval gating resets

                        # ensure channel
                        out_cid = ensure_channel(channel.id)
                        ch = summary.channels.get(channel.id)
                        if topics is not None and ch and ch.topic not in topics:
                            continue

                        # Decode schema minimally to find "format" for foxglove.CompressedVideo
                        # We don't need full protobuf decode; infer based on topic's recent frames.
                        fmt = None
                        try:
                            # Heuristic: foxglove.CompressedVideo messages tend to be > 100 bytes if compressed.
                            # We'll look at last stored codec for the topic and update from payload if needed.
                            pass
                        except Exception:
                            pass

                        # Determine codec and update SPS/PPS cache if H26x
                        topic = ch.topic if ch else f"/unknown/{channel.id}"
                        data = message.data
                        codec = last_codec.get(topic)
                        # Try to detect format by looking for NAL start codes
                        if data.startswith(_START3) or data.startswith(_START4):
                            # Could be H264 or H265; detect by presence of typical NAL types
                            # Prefer to keep previous guess unless unknown
                            if codec not in ("h264", "hevc"):
                                # naive: look for h264 SPS (type 7) quickly
                                is264 = False
                                for _, n in _split_nalus(data):
                                    tpy = _h264_type(n)
                                    if tpy in (7, 8, 5):
                                        is264 = True
                                        break
                                codec = "h264" if is264 else "hevc"
                                last_codec[topic] = codec

                            if codec == "h264":
                                blob = _collect_sps_pps_h264(data)
                                if blob:
                                    last_ppsblob[topic] = blob
                                is_idr = _is_h264_idr(data)
                            else:
                                blob = _collect_vps_sps_pps_h265(data)
                                if blob:
                                    last_ppsblob[topic] = blob
                                is_idr = _is_h265_idr(data)
                        else:
                            # jpeg or other
                            codec = codec or "jpeg"
                            last_codec[topic] = codec
                            is_idr = True  # JPEG frames are intra

                        # Keyframe gating per topic within each interval
                        if args.start_on_keyframe:
                            if current_interval_start_ns is not None:
                                if topic not in gating_on_key:
                                    gating_on_key[topic] = True  # start gating at interval entry
                                if gating_on_key[topic]:
                                    # Need SPS/PPS available and an IDR for H26x; JPEG passes immediately
                                    if last_codec.get(topic) in ("h264", "hevc"):
                                        have_hdr = topic in last_ppsblob
                                        if have_hdr and is_idr:
                                            # prefix SPS/PPS before the first written access unit
                                            # We do this by emitting an extra message at same timestamp containing headers,
                                            # which many decoders accept as a parameter-only sample.
                                            hdr = last_ppsblob[topic]
                                            # write header message (same timestamps)
                                            out_log = message.log_time if args.preserve_times else (concat_base_ns + (t - current_interval_start_ns))
                                            out_pub = out_log
                                            writer.add_message(channel_id=out_cid, log_time=out_log, publish_time=out_pub, data=hdr)
                                            gating_on_key[topic] = False
                                        else:
                                            # keep buffering until we see headers and IDR; skip this frame
                                            continue
                                    else:
                                        gating_on_key[topic] = False  # JPEG/other — no gating needed

                        if args.preserve_times:
                            out_log = message.log_time
                            out_pub = message.publish_time
                        else:
                            delta = (t - current_interval_start_ns) if current_interval_start_ns is not None else 0
                            out_log = concat_base_ns + delta
                            out_pub = out_log

                        writer.add_message(
                            channel_id=out_cid,
                            log_time=out_log,
                            publish_time=out_pub,
                            data=message.data,
                        )

                    # close interval gap if needed
                    if not args.preserve_times and current_interval_start_ns is not None:
                        concat_base_ns += gap_ns
                        current_interval_start_ns = None

        finally:
            writer.finish()

    # quick validation
    with open(out_path, "rb") as _vf:
        make_reader(_vf)
    print(f"[OK] Merged spliced MCAP → {out_path}")

if __name__ == "__main__":
    main()
