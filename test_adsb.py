#!/usr/bin/env python3
"""Quick test of ADS-B MCAP reading and GeoJSON generation."""

import os
import sys
import json
from adsb_geojson_streamer import ADSBDataReader, time_sync_adsb_to_video

def main():
    adsb_path = os.path.expanduser("~/Downloads/adsb-decodes/20251024-181136-adsb-decode.mcap")
    video_path = os.path.expanduser("~/Downloads/merged_spliced.mcap")
    
    if not os.path.exists(adsb_path):
        print(f"ADS-B MCAP not found: {adsb_path}")
        return 1
    
    print("=" * 60)
    print("Loading ADS-B data...")
    print("=" * 60)
    
    reader = ADSBDataReader(adsb_path)
    print(f"Loaded {len(reader.messages)} messages")
    
    if reader.messages:
        first_ts, first_msg = reader.messages[0]
        last_ts, last_msg = reader.messages[-1]
        print(f"First message: {first_ts / 1e9:.1f} s UTC")
        print(f"Last message:  {last_ts / 1e9:.1f} s UTC")
        print(f"Duration: {(last_ts - first_ts) / 1e9 / 3600:.1f} hours")
        print(f"\nFirst message content:")
        print(json.dumps(first_msg, indent=2)[:500])
    
    # Test time sync
    if os.path.exists(video_path):
        print("\n" + "=" * 60)
        print("Time sync test...")
        print("=" * 60)
        time_sync_adsb_to_video(adsb_path, video_path, video_start_offset_secs=145)
    
    # Test aircraft extraction
    if reader.messages:
        print("\n" + "=" * 60)
        print("Aircraft extraction test...")
        print("=" * 60)
        test_ts = reader.messages[100][0] if len(reader.messages) > 100 else reader.messages[0][0]
        aircraft = reader.get_aircraft_at_time(test_ts, lookback_secs=60)
        print(f"Found {len(aircraft)} aircraft at timestamp {test_ts / 1e9:.1f}")
        for icao, state in list(aircraft.items())[:3]:
            print(f"  {icao}: {state.callsign or 'N/A'} @ ({state.lat:.4f}, {state.lon:.4f}) ALT={state.altitude_m}m")
    
    print("\n✓ All tests passed!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
