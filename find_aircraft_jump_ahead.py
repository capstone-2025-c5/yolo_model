from adsb_geojson_streamer import ADSBDataReader

reader = ADSBDataReader("/Users/jonah/Downloads/adsb-decodes/20251025-103624-adsb-decode.mcap")

print("Finding when ADS-B data resumes after gap...")
print()

# Gap is from message 1926 (1761407069.13s) to 1927 (1761410955.96s)
gap_end_msg_idx = 1927
gap_end_time = reader.messages[gap_end_msg_idx][0] / 1e9

print(f"ADS-B gap ends at message {gap_end_msg_idx}: {gap_end_time:.2f}s")
print()

# Calculate offset needed for this time
video_start = 1761403518.8
offset_to_gap_end = gap_end_time - video_start
print(f"Video offset to reach gap end: {offset_to_gap_end:.1f}s ({int(offset_to_gap_end/60)}m {int(offset_to_gap_end%60)}s)")
print()

# Now search through messages after the gap for aircraft on approach/landing
# Look for aircraft with low altitude and descending track
print("Searching for aircraft on final approach (low altitude, descending)...")
print()

landing_candidates = {}  # icao -> list of states

for msg_idx in range(gap_end_msg_idx, len(reader.messages)):
    ts_ns, aircraft_list = reader.messages[msg_idx]
    ts_s = ts_ns / 1e9
    
    if isinstance(aircraft_list, list):
        for aircraft in aircraft_list:
            if isinstance(aircraft, dict):                
                icao = aircraft.get("hex", "").upper()
                call = (aircraft.get("flight") or "").strip()
                alt = aircraft.get("alt_baro")
                lat = aircraft.get("lat")
                lon = aircraft.get("lon")
                track = aircraft.get("track")
                spd = aircraft.get("gs")
                
                # Look for aircraft with valid position and low altitude
                if lat and lon and alt and spd and track is not None:
                    # Convert altitude (could be numeric or "ground")
                    if isinstance(alt, str):
                        if alt.lower() == "ground":
                            alt_m = 0
                        else:
                            continue
                    else:
                        alt_m = alt
                    
                    # Aircraft on approach: altitude < 3000 feet, descending or landing
                    if alt_m < 3000 * 0.3048:  # Convert to meters
                        if icao not in landing_candidates:
                            landing_candidates[icao] = []
                        landing_candidates[icao].append({
                            "ts": ts_s,
                            "call": call,
                            "alt": alt_m,
                            "lat": lat,
                            "lon": lon,
                            "track": track,
                            "spd": spd
                        })

# Find the best candidates (aircraft with multiple low-altitude messages)
print("Aircraft on approach (multiple observations):")
best_candidates = [(icao, states) for icao, states in landing_candidates.items() if len(states) >= 5]
best_candidates.sort(key=lambda x: -len(x[1]))

for icao, states in best_candidates[:10]:
    call = states[0]["call"]
    first_ts = states[0]["ts"]
    first_offset = first_ts - video_start
    alt_range = f"{min(s['alt'] for s in states):.0f}-{max(s['alt'] for s in states):.0f}m"
    print(f"  {icao} ({call}): {len(states)} observations at offset {first_offset:.0f}s, alt {alt_range}")
    
    # Show first few observations
    for i, state in enumerate(states[:3]):
        print(f"    [{i}] {state['ts']:.1f}s alt={state['alt']:.0f}m spd={state['spd']:.0f}kt")
        