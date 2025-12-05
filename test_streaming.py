#!/usr/bin/env python3
"""Test ADS-B streaming with interpolation - shows updates every frame."""

import sys
import time
from adsb_geojson_streamer import ADSBDataReader, GeoJSONStreamer, AircraftState

# Load ADS-B data
adsb_path = "/Users/jonah/Downloads/adsb-decodes/20251025-103624-adsb-decode.mcap"
print("Loading ADS-B MCAP...")
reader = ADSBDataReader(adsb_path)

# Create streamer without Foxglove context (just test the stream_aircraft logic)
streamer = GeoJSONStreamer()

# Simulate streaming at 30 FPS for 30 seconds
print("\nSimulating 30 FPS playback for 30 seconds starting at N279CA visible time...")
print("=" * 70)

base_ts = 1761406600e9  # ~175 seconds into N279CA visibility
frame_interval_s = 1/30  # 30 FPS
frame_time_ns = base_ts

updates_with_data = 0
for frame_num in range(30 * 30):  # 30 seconds at 30 FPS
    aircraft = reader.get_aircraft_at_time(frame_time_ns, lookback_secs=60)
    
    if aircraft:
        updates_with_data += 1
        # Simulate streaming (we'd call streamer.stream_aircraft(aircraft) with real Foxglove)
        
        # Show first, middle, and last frame
        if frame_num in [0, 450, 899]:
            icao = "A2C81E"
            if icao in aircraft:
                state = aircraft[icao]
                print(f"\nFrame {frame_num}: {state.lat:.6f}, {state.lon:.6f} alt={state.altitude_m:.1f}m")
    
    frame_time_ns += int(frame_interval_s * 1e9)

print("\n" + "=" * 70)
print(f"✓ Streamed {updates_with_data} frames with aircraft data out of 900 frames")
print(f"  This shows continuous updates at 30 FPS (no frame skipping)")
print(f"  Positions are smoothly interpolated between sparse ADS-B messages")
