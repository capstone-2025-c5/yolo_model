#!/usr/bin/env python3
"""Debug script to check ADS-B data for specific aircraft."""

import sys
from adsb_geojson_streamer import ADSBDataReader

# Load ADS-B data
reader = ADSBDataReader("/Users/jonah/Downloads/adsb-decodes/20251025-103624-adsb-decode.mcap")

print(f"Loaded {len(reader.messages)} ADS-B messages")
if reader.messages:
    first_ts = reader.messages[0][0] / 1e9
    last_ts = reader.messages[-1][0] / 1e9
    print(f"ADS-B time range: {first_ts:.2f} - {last_ts:.2f} s")
print()

# Get all aircraft
all_icaos = reader.get_aircraft_icaos()
print(f"Found {len(all_icaos)} unique aircraft ICAOs")
print()

# Check for N279CA by callsign instead
test_callsign = "N279CA"
print(f"Checking for aircraft with callsign: {test_callsign}")

# Search through all messages for this callsign
matching_data = []
for ts_ns, aircraft_list in reader.messages:
    if isinstance(aircraft_list, list):
        for aircraft in aircraft_list:
            if isinstance(aircraft, dict):
                call = (aircraft.get("flight") or "").strip()
                if call == test_callsign:
                    ts_s = ts_ns / 1e9
                    icao = aircraft.get("hex", "").upper()
                    matching_data.append({
                        "timestamp_s": ts_s,
                        "icao": icao,
                        "data": aircraft
                    })

print(f"Found {len(matching_data)} messages for callsign {test_callsign}")
print()

if matching_data:
    print("First 5 messages:")
    for i, msg in enumerate(matching_data[:5]):
        ts_s = msg["timestamp_s"]
        icao = msg["icao"]
        aircraft = msg["data"]
        lat = aircraft.get("lat")
        lon = aircraft.get("lon")
        alt = aircraft.get("alt_baro")
        spd = aircraft.get("gs")
        track = aircraft.get("track")
        call = aircraft.get("flight", "").strip()
        print(f"  [{i}] {ts_s:.2f}s ICAO={icao}: lat={lat}, lon={lon}, alt={alt}, spd={spd}, track={track}, call={call}")
    
    print()
    print("Last 5 messages:")
    for i, msg in enumerate(matching_data[-5:]):
        ts_s = msg["timestamp_s"]
        icao = msg["icao"]
        aircraft = msg["data"]
        lat = aircraft.get("lat")
        lon = aircraft.get("lon")
        alt = aircraft.get("alt_baro")
        spd = aircraft.get("gs")
        track = aircraft.get("track")
        call = aircraft.get("flight", "").strip()
        idx = len(matching_data) - 5 + i
        print(f"  [{idx}] {ts_s:.2f}s ICAO={icao}: lat={lat}, lon={lon}, alt={alt}, spd={spd}, track={track}, call={call}")
else:
    print(f"No data found for callsign {test_callsign}")
    print()
    print("Sample aircraft with their callsigns:")
    count = 0
    for icao in sorted(all_icaos)[:50]:
        # Get one message to check callsign
        msg_list = reader.get_aircraft_data_for_icao(icao, limit=1)
        if msg_list and msg_list[0]["data"]:
            call = msg_list[0]["data"].get("flight", "").strip()
            if call:
                print(f"  {icao}: {call}")
                count += 1
                if count >= 15:
                    break
