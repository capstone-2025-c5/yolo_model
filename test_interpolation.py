#!/usr/bin/env python3
"""Test interpolation of aircraft positions between ADS-B messages."""

import sys
from adsb_geojson_streamer import ADSBDataReader

# Load ADS-B data
adsb_path = "/Users/jonah/Downloads/adsb-decodes/20251025-103624-adsb-decode.mcap"
print("Loading ADS-B MCAP...")
reader = ADSBDataReader(adsb_path)

# Get N279CA (ICAO: A2C81E) timeline
print("\nAnalyzing N279CA (A2C81E) interpolation...")
print("=" * 70)

# Query at 5 different times within a 30-second span to show interpolation
base_ts = 1761406600e9  # ~175 seconds into N279CA visibility

for i in range(6):
    query_ts = base_ts + i * 5e9  # Query every 5 seconds
    
    aircraft = reader.get_aircraft_at_time(query_ts, lookback_secs=60)
    
    if "A2C81E" in aircraft:
        state = aircraft["A2C81E"]
        print(f"\nTime offset +{i*5}s:")
        print(f"  Position: {state.lat:.6f}, {state.lon:.6f}")
        print(f"  Altitude: {state.altitude_m:.1f}m" if state.altitude_m else "  Altitude: None")
        print(f"  Track: {state.track_degrees:.1f}°" if state.track_degrees else "  Track: None")
        print(f"  Speed: {state.ground_speed_knots:.1f} kt" if state.ground_speed_knots else "  Speed: None")
    else:
        print(f"\nTime offset +{i*5}s: No aircraft data")

print("\n" + "=" * 70)
print("✓ Interpolation test complete - positions should show gradual change")
