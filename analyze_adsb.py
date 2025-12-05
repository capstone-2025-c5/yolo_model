#!/usr/bin/env python3
"""
Analyze and compare YOLO detections with ADS-B aircraft positions.

This utility can:
1. List all aircraft in ADS-B MCAP during video duration
2. Show aircraft track over time
3. Export detections to CSV
"""

import argparse
import csv
import os
from datetime import datetime
from adsb_geojson_streamer import ADSBDataReader, time_sync_adsb_to_video

def cmd_list_aircraft(args):
    """List all aircraft detected in ADS-B MCAP."""
    print(f"Loading ADS-B data: {args.adsb_mcap}")
    reader = ADSBDataReader(args.adsb_mcap)
    
    # Collect all unique aircraft
    all_aircraft = {}
    for ts, aircraft_list in reader.messages:
        for ac in aircraft_list:
            icao = ac.get("hex", "UNKNOWN").upper()
            if icao not in all_aircraft:
                all_aircraft[icao] = {
                    "icao": icao,
                    "callsign": ac.get("flight", "").strip() or None,
                    "first_seen": ts,
                    "last_seen": ts,
                    "positions": 0,
                    "with_position": 0,
                }
            rec = all_aircraft[icao]
            rec["last_seen"] = max(rec["last_seen"], ts)
            rec["positions"] += 1
            if ac.get("lat") is not None and ac.get("lon") is not None:
                rec["with_position"] += 1
    
    print(f"\nFound {len(all_aircraft)} unique aircraft")
    print()
    
    # Sort by first seen
    sorted_ac = sorted(all_aircraft.values(), key=lambda x: x["first_seen"])
    
    print(f"{'ICAO':<8} {'Callsign':<12} {'First Seen (UTC)':<20} {'Last Seen':<10} {'Updates':<10} {'With Pos':<10}")
    print("-" * 80)
    
    for ac in sorted_ac:
        first_dt = datetime.utcfromtimestamp(ac["first_seen"] / 1e9).strftime("%Y-%m-%d %H:%M:%S")
        last_dt = datetime.utcfromtimestamp(ac["last_seen"] / 1e9).strftime("%H:%M:%S")
        callsign = ac["callsign"] or "-"
        print(f"{ac['icao']:<8} {callsign:<12} {first_dt:<20} {last_dt:<10} {ac['positions']:<10} {ac['with_position']:<10}")

def cmd_track(args):
    """Export aircraft track for a specific ICAO code."""
    print(f"Loading ADS-B data: {args.adsb_mcap}")
    reader = ADSBDataReader(args.adsb_mcap)
    
    icao = args.icao.upper()
    track = []
    
    for ts, aircraft_list in reader.messages:
        for ac in aircraft_list:
            if ac.get("hex", "").upper() == icao:
                if ac.get("lat") is not None and ac.get("lon") is not None:
                    track.append({
                        "timestamp": ts / 1e9,
                        "datetime": datetime.utcfromtimestamp(ts / 1e9).isoformat(),
                        "lat": ac.get("lat"),
                        "lon": ac.get("lon"),
                        "altitude": ac.get("alt_geom") or ac.get("alt_baro"),
                        "track": ac.get("track"),
                        "speed": ac.get("gs"),
                        "callsign": ac.get("flight", "").strip() or None,
                    })
    
    if not track:
        print(f"No track data for {icao}")
        return
    
    print(f"\nAircraft {icao}: {len(track)} position reports")
    print()
    
    # Export to CSV if requested
    if args.output:
        with open(args.output, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=track[0].keys())
            writer.writeheader()
            writer.writerows(track)
        print(f"Exported to: {args.output}")
    else:
        # Print summary
        for i, pt in enumerate(track[:5]):
            print(f"  {pt['datetime']}: ({pt['lat']:.4f}, {pt['lon']:.4f}) "
                  f"ALT={pt['altitude']:.0f}m TRK={pt['track']:.0f}° SPD={pt['speed']:.0f}kt")
        if len(track) > 5:
            print(f"  ... ({len(track) - 5} more)")

def cmd_summary(args):
    """Show time sync summary."""
    adsb_path = args.adsb_mcap
    video_path = args.video_mcap
    
    print("Time Synchronization Summary")
    print("=" * 60)
    time_sync_adsb_to_video(
        adsb_path,
        video_path,
        video_start_offset_secs=args.start_offset,
        output_path="adsb_sync.json" if args.save_config else None,
    )

def main():
    parser = argparse.ArgumentParser(description="ADS-B data analysis utility")
    parser.add_argument("--adsb-mcap", required=True, help="Path to ADS-B MCAP file")
    parser.add_argument("--video-mcap", help="Path to video MCAP file (for time sync)")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # list command
    subparsers.add_parser("list", help="List all aircraft in ADS-B file")
    
    # track command
    track_parser = subparsers.add_parser("track", help="Export track for aircraft ICAO")
    track_parser.add_argument("icao", help="ICAO code (6 hex digits)")
    track_parser.add_argument("-o", "--output", help="Export to CSV file")
    
    # sync command
    sync_parser = subparsers.add_parser("sync", help="Show time sync with video")
    sync_parser.add_argument("--start-offset", type=float, default=0, help="Video start offset (seconds)")
    sync_parser.add_argument("--save-config", action="store_true", help="Save sync config to adsb_sync.json")
    
    args = parser.parse_args()
    
    if args.command == "list":
        cmd_list_aircraft(args)
    elif args.command == "track":
        cmd_track(args)
    elif args.command == "sync":
        if not args.video_mcap:
            print("Error: --video-mcap required for sync command")
            return 1
        cmd_summary(args)
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
