#!/usr/bin/env python3
from adsb_geojson_streamer import ADSBDataReader

reader = ADSBDataReader("/Users/jonah/Downloads/adsb-decodes/20251025-103624-adsb-decode.mcap")

print(f"Total messages: {len(reader.messages)}")

if reader.messages:
    first_ts = reader.messages[0][0] / 1e9
    last_ts = reader.messages[-1][0] / 1e9
    print(f"Time range: {first_ts:.2f}s - {last_ts:.2f}s")
    print(f"Duration: {(last_ts - first_ts):.1f}s")
    
    # Check around 1761408890
    target = 1761408890.52
    print(f"\nLooking for data around {target:.2f}s...")
    
    if target >= first_ts and target <= last_ts:
        print(f"  Target is within range")
        # Find closest
        found = 0
        for i, (ts_ns, aircraft_list) in enumerate(reader.messages):
            ts_s = ts_ns / 1e9
            if ts_s >= target - 10 and ts_s <= target + 10:
                num = len(aircraft_list) if isinstance(aircraft_list, list) else 0
                print(f"    [{i}] {ts_s:.2f}s: {num} aircraft (type={type(aircraft_list).__name__})")
                found += 1
                if found >= 15:
                    break
        if found == 0:
            print("    No messages found within ±10s window!")
            print(f"\n  Message distribution analysis:")
            # Check how messages are spaced
            prev_ts = 0
            gaps = []
            for i, (ts_ns, _) in enumerate(reader.messages):
                ts_s = ts_ns / 1e9
                if i > 0:
                    gap = ts_s - prev_ts
                    gaps.append(gap)
                prev_ts = ts_s
            
            if gaps:
                import statistics
                print(f"    Gap statistics: min={min(gaps):.2f}s, max={max(gaps):.2f}s, avg={statistics.mean(gaps):.2f}s")
            
            # Find where 1761408890 falls in the message sequence
            print(f"\n  Checking position of {target:.2f}s in message timeline...")
            for i, (ts_ns, _) in enumerate(reader.messages):
                ts_s = ts_ns / 1e9
                if ts_s > target:
                    print(f"    Message {i-1}: {reader.messages[i-1][0]/1e9:.2f}s")
                    print(f"    Target:     {target:.2f}s")
                    print(f"    Message {i}: {ts_s:.2f}s")
                    break
    else:
        print(f"  Target {target:.2f}s is OUTSIDE range {first_ts:.2f}-{last_ts:.2f}s")
