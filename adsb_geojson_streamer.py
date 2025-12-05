"""
ADS-B aircraft data streamer for Foxglove visualization as GeoJSON.

Reads ADS-B position data from MCAP files, time-syncs to video MCAP,
and streams as GeoJSON FeatureCollection to Foxglove.
"""

import json
import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import struct

from mcap.stream_reader import StreamReader
from foxglove.schemas import GeoJson

logger = logging.getLogger(__name__)


@dataclass
class AircraftState:
    """Current state of an aircraft from ADS-B data."""
    icao: str  # 6-hex digit ICAO code
    callsign: Optional[str]  # Flight identifier
    lat: Optional[float]
    lon: Optional[float]
    altitude_m: Optional[float]
    track_degrees: Optional[float]
    ground_speed_knots: Optional[float]
    timestamp: float  # Nanoseconds since epoch


class ADSBDataReader:
    """Read and parse ADS-B data from MCAP files."""
    
    def __init__(self, mcap_path: str):
        """
        Args:
            mcap_path: Path to ADS-B MCAP file
        """
        self.mcap_path = Path(mcap_path)
        self.aircraft_states: Dict[str, AircraftState] = {}
        self.messages: List[Tuple[float, List[Dict]]] = []  # (timestamp_ns, [aircraft_dict, ...])
        self._load_adsb_data()
        
    def _load_adsb_data(self) -> None:
        """Load and parse all ADS-B messages from MCAP."""
        logger.info(f"Loading ADS-B data from {self.mcap_path}")
        
        with open(self.mcap_path, "rb") as f:
            reader = StreamReader(f)
            
            for record in reader.records:
                # Parse messages
                if hasattr(record, "channel_id") and hasattr(record, "data"):
                    ts_ns = record.log_time
                    message_bytes = record.data
                    
                    try:
                        # foxglove.Log protobuf contains JSON array of aircraft
                        # Extract JSON from protobuf-encoded message
                        msg_dict = self._parse_log_message(message_bytes)
                        if msg_dict:
                            self.messages.append((ts_ns, msg_dict))
                    except Exception as e:
                        logger.debug(f"Failed to parse message: {e}")
        
        logger.info(f"Loaded {len(self.messages)} ADS-B messages")
        # Sort by timestamp
        self.messages.sort(key=lambda x: x[0])
    
    def _parse_altitude(self, alt_value) -> Optional[float]:
        """
        Parse altitude value, handling numeric and string formats.
        Returns altitude in meters, or None if invalid.
        """
        if alt_value is None:
            return None
        if isinstance(alt_value, str):
            # Handle special cases like "ground"
            if alt_value.lower() in ("ground", "none"):
                return 0.0
            try:
                return float(alt_value)
            except (ValueError, TypeError):
                return None
        if isinstance(alt_value, (int, float)):
            return float(alt_value)
        return None
        
    def _parse_log_message(self, data: bytes) -> Optional[List[Dict]]:
        """
        Parse foxglove.Log protobuf message containing JSON array of aircraft.
        Returns list of aircraft dictionaries with lat/lon/icao/etc.
        """
        try:
            # foxglove.Log contains protobuf-encoded text field with JSON
            # Find JSON array in the protobuf data
            json_start = data.find(b'[{')
            if json_start < 0:
                return None
            
            json_end = data.rfind(b'}]')
            if json_end < 0:
                return None
            
            json_end += 2  # Include the closing }]
            json_text = data[json_start:json_end].decode("utf-8", errors="ignore")
            
            if not json_text.startswith("["):
                return None
            
            return json.loads(json_text)
        except Exception as e:
            logger.debug(f"Failed to parse log message: {e}")
            return None
    
    def get_aircraft_icaos(self) -> set:
        """Get all unique ICAO codes in the ADS-B data."""
        icaos = set()
        for ts_ns, aircraft_list in self.messages:
            if isinstance(aircraft_list, list):
                for aircraft in aircraft_list:
                    if isinstance(aircraft, dict):
                        icao = aircraft.get("hex", "").upper()
                        if icao:
                            icaos.add(icao)
        return icaos
    
    def get_aircraft_data_for_icao(self, icao: str, limit: int = 20) -> list:
        """Get all messages for a specific aircraft ICAO code."""
        icao = icao.upper()
        data = []
        for ts_ns, aircraft_list in self.messages:
            if isinstance(aircraft_list, list):
                for aircraft in aircraft_list:
                    if isinstance(aircraft, dict):
                        if aircraft.get("hex", "").upper() == icao:
                            ts_s = ts_ns / 1e9
                            data.append({"timestamp_s": ts_s, "data": aircraft})
                            if len(data) >= limit:
                                return data
        return data
    
    def get_aircraft_at_time(self, timestamp_ns: float, lookback_secs: float = 60) -> Dict[str, AircraftState]:
        """
        Get aircraft positions at given timestamp, interpolating between ADS-B messages.
        
        For each aircraft, finds the two closest ADS-B messages (before and after the target time)
        and interpolates position between them.
        
        Args:
            timestamp_ns: Target timestamp in nanoseconds
            lookback_secs: How far back to look for data (defines max age of earlier message)
            
        Returns:
            Dict mapping ICAO code to AircraftState with interpolated positions
        """
        lookback_ns = int(lookback_secs * 1e9)
        window_start = timestamp_ns - lookback_ns
        
        # Build timeline of all aircraft states
        aircraft_timeline = {}  # icao -> [(ts_ns, state), (ts_ns, state), ...]
        
        for ts_ns, aircraft_list in self.messages:
            if ts_ns < window_start:
                continue
                
            if isinstance(aircraft_list, list):
                for aircraft in aircraft_list:
                    if isinstance(aircraft, dict):
                        icao = aircraft.get("hex", "UNKNOWN").upper()
                        if not icao or icao == "UNKNOWN":
                            continue
                        
                        lat = aircraft.get("lat")
                        lon = aircraft.get("lon")
                        
                        # Skip if no position
                        if lat is None or lon is None:
                            continue
                        
                        state = AircraftState(
                            icao=icao,
                            callsign=aircraft.get("flight", "").strip() if aircraft.get("flight") else None,
                            lat=lat,
                            lon=lon,
                            altitude_m=self._parse_altitude(aircraft.get("alt_geom") or aircraft.get("alt_baro")),
                            track_degrees=aircraft.get("track"),
                            ground_speed_knots=aircraft.get("gs"),
                            timestamp=ts_ns,
                        )
                        
                        if icao not in aircraft_timeline:
                            aircraft_timeline[icao] = []
                        aircraft_timeline[icao].append((ts_ns, state))
        
        # Now interpolate to target time
        result = {}
        for icao, timeline in aircraft_timeline.items():
            # Find the two closest states (before and after target time)
            before_state: Optional[AircraftState] = None
            after_state: Optional[AircraftState] = None
            before_ts_ns: float = 0.0
            after_ts_ns: float = 0.0
            
            for ts_ns, state in timeline:
                if ts_ns <= timestamp_ns:
                    before_state = state
                    before_ts_ns = float(ts_ns)
                else:
                    after_state = state
                    after_ts_ns = float(ts_ns)
                    break
            
            # Use the state we have (prefer interpolation, fallback to before/after)
            if before_state and after_state and after_ts_ns > 0:
                # Interpolate between before and after
                interpolated = self._interpolate_aircraft_state(
                    before_state, before_ts_ns,
                    after_state, after_ts_ns,
                    timestamp_ns
                )
                result[icao] = interpolated
            elif before_state:
                # Use the latest known state before target time
                result[icao] = before_state
            elif after_state:
                # Use the next state if we don't have a before state
                result[icao] = after_state
        
        return result
    
    def _interpolate_aircraft_state(
        self,
        state1: AircraftState,
        ts1_ns: float,
        state2: AircraftState,
        ts2_ns: float,
        target_ts_ns: float,
    ) -> AircraftState:
        """
        Interpolate aircraft state between two measurements.
        
        Args:
            state1: Aircraft state at earlier time
            ts1_ns: Timestamp of state1 in nanoseconds
            state2: Aircraft state at later time
            ts2_ns: Timestamp of state2 in nanoseconds
            target_ts_ns: Target timestamp to interpolate to
            
        Returns:
            Interpolated AircraftState at target time
        """
        # Time fraction (0 to 1)
        time_fraction = (target_ts_ns - ts1_ns) / (ts2_ns - ts1_ns)
        time_fraction = max(0.0, min(1.0, time_fraction))  # Clamp to [0, 1]
        
        # Interpolate position
        lat = state1.lat + (state2.lat - state1.lat) * time_fraction if state1.lat and state2.lat else state1.lat
        lon = state1.lon + (state2.lon - state1.lon) * time_fraction if state1.lon and state2.lon else state1.lon
        
        # Interpolate altitude (or use earlier if string)
        alt1 = state1.altitude_m or 0
        alt2 = state2.altitude_m or 0
        altitude = alt1 + (alt2 - alt1) * time_fraction
        
        # Use earlier state's callsign (flights don't change), but average track/speed
        track = None
        if state1.track_degrees and state2.track_degrees:
            track = state1.track_degrees + (state2.track_degrees - state1.track_degrees) * time_fraction
        elif state1.track_degrees:
            track = state1.track_degrees
        elif state2.track_degrees:
            track = state2.track_degrees
        
        speed = None
        if state1.ground_speed_knots and state2.ground_speed_knots:
            speed = state1.ground_speed_knots + (state2.ground_speed_knots - state1.ground_speed_knots) * time_fraction
        elif state1.ground_speed_knots:
            speed = state1.ground_speed_knots
        elif state2.ground_speed_knots:
            speed = state2.ground_speed_knots
        
        return AircraftState(
            icao=state1.icao,
            callsign=state1.callsign,
            lat=lat,
            lon=lon,
            altitude_m=altitude,
            track_degrees=track,
            ground_speed_knots=speed,
            timestamp=target_ts_ns,
        )


class GeoJSONStreamer:
    """Stream aircraft positions as GeoJSON to Foxglove."""
    
    def __init__(self, topic: str = "/aircraft/positions", context=None):
        """
        Args:
            topic: Foxglove topic for GeoJSON output
            context: Foxglove context
        """
        self.topic = topic
        self.context = context
        self._channel = None
        self._last_published_state = None  # Track last published aircraft state
        self._last_frame_time_ns = None    # Track last frame time for jump detection
        self._initialize_channel()
        
    def _initialize_channel(self) -> None:
        """Create Foxglove channel for GeoJSON."""
        if self._channel is None and self.context is not None:
            try:
                from foxglove.channels import GeoJsonChannel
                
                # Use foxglove GeoJsonChannel
                self._channel = GeoJsonChannel(
                    topic=self.topic,
                    context=self.context,
                )
                logger.info(f"Initialized Foxglove GeoJSON channel: {self.topic}")
            except Exception as e:
                logger.error(f"Failed to create GeoJSON channel: {e}")
    
    def stream_aircraft(self, aircraft_states: Dict[str, AircraftState]) -> None:
        """
        Stream aircraft positions as GeoJSON FeatureCollection.
        Publishes on every frame to show smooth interpolated motion between ADS-B updates.
        
        Args:
            aircraft_states: Dict mapping ICAO code to AircraftState
        """
        if self._channel is None:
            return
        
        if not aircraft_states:
            logger.debug("No aircraft states to stream")
            return
        
        # Detect if video jumped ahead (time discontinuity)
        current_time = None
        for state in aircraft_states.values():
            if state.timestamp is not None:
                current_time = state.timestamp
                break
        
        if current_time is not None:
            self._last_frame_time_ns = current_time
        
        features = []
        for icao, state in aircraft_states.items():
            if state.lat is None or state.lon is None:
                continue
            
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [state.lon, state.lat],  # GeoJSON: [lon, lat]
                },
                "properties": {
                    "name": state.callsign or f"Aircraft {icao}",
                    "icao": icao,
                    "callsign": state.callsign or "Unknown",
                    "altitude_m": state.altitude_m or 0,
                    "track_deg": state.track_degrees or 0,
                    "speed_knots": state.ground_speed_knots or 0,
                    "timestamp": state.timestamp,
                    "style": {
                        "color": "#00ff00",  # Green for aircraft
                        "opacity": "0.8",
                        "weight": 2,
                    },
                },
            }
            features.append(feature)
        
        geojson_obj = {
            "type": "FeatureCollection",
            "features": features,
        }
        
        try:
            # GeoJsonChannel expects a GeoJson protobuf with geojson string containing JSON
            msg = GeoJson(geojson=json.dumps(geojson_obj))
            self._channel.log(msg)
        except Exception as e:
            logger.error(f"Failed to stream GeoJSON: {e}")
            # Log aircraft states for debugging format errors
            for icao, state in aircraft_states.items():
                logger.error(f"  Aircraft {icao}: lat={state.lat} ({type(state.lat).__name__}), lon={state.lon} ({type(state.lon).__name__}), alt={state.altitude_m} ({type(state.altitude_m).__name__}), speed={state.ground_speed_knots} ({type(state.ground_speed_knots).__name__})")
    
    def _create_state_signature(self, aircraft_states: Dict[str, AircraftState]) -> str:
        """
        Create a signature of the aircraft state for change detection.
        Signature includes ICAO, lat/lon, altitude, and speed.
        """
        sig_parts = []
        for icao in sorted(aircraft_states.keys()):
            state = aircraft_states[icao]
            # Handle None values safely
            lat = state.lat if state.lat is not None else 0.0
            lon = state.lon if state.lon is not None else 0.0
            alt = state.altitude_m if state.altitude_m is not None else 0.0
            speed = state.ground_speed_knots if state.ground_speed_knots is not None else 0.0
            # Round to reasonable precision to avoid floating point noise
            sig = f"{icao}:{lat:.2f},{lon:.2f},{alt:.0f},{speed:.1f}"
            sig_parts.append(sig)
        return "|".join(sig_parts)


def time_sync_adsb_to_video(
    adsb_mcap: str,
    video_mcap: str,
    video_start_offset_secs: float = 0,
    output_path: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Determine time synchronization between ADS-B and video MCAPs.
    
    Args:
        adsb_mcap: Path to ADS-B MCAP
        video_mcap: Path to video MCAP
        video_start_offset_secs: Offset to start reading video (from --start-offset)
        output_path: Optional path to save sync config
        
    Returns:
        (adsb_start_ns, video_start_ns): Synchronized start timestamps in nanoseconds
    """
    # Get MCAP timestamps
    adsb_start, adsb_end = _get_mcap_time_range(adsb_mcap)
    video_start, video_end = _get_mcap_time_range(video_mcap)
    
    # Apply video offset
    video_start_ns = video_start + int(video_start_offset_secs * 1e9)
    
    logger.info(f"ADS-B range:  {adsb_start / 1e9:.1f} - {adsb_end / 1e9:.1f} s")
    logger.info(f"Video range:  {video_start / 1e9:.1f} - {video_end / 1e9:.1f} s")
    logger.info(f"Video start (with offset): {video_start_ns / 1e9:.1f} s")
    
    # Check for overlap
    overlap_start = max(adsb_start, video_start_ns)
    overlap_end = min(adsb_end, video_end)
    
    if overlap_start >= overlap_end:
        logger.warning(f"No time overlap between datasets!")
        logger.warning(f"  ADS-B: {adsb_start / 1e9:.1f} - {adsb_end / 1e9:.1f} s")
        logger.warning(f"  Video: {video_start_ns / 1e9:.1f} - {video_end / 1e9:.1f} s")
    else:
        overlap_secs = (overlap_end - overlap_start) / 1e9
        logger.info(f"Overlap: {overlap_secs:.1f} seconds")
    
    if output_path:
        sync_config = {
            "adsb_start_ns": adsb_start,
            "video_start_ns": video_start_ns,
            "overlap_start_ns": overlap_start,
            "overlap_end_ns": overlap_end,
        }
        with open(output_path, "w") as f:
            json.dump(sync_config, f, indent=2)
        logger.info(f"Saved sync config to {output_path}")
    
    return adsb_start, video_start_ns


def _get_mcap_time_range(mcap_path: str) -> Tuple[float, float]:
    """Get start and end timestamps from MCAP file (in nanoseconds)."""
    min_ts = float("inf")
    max_ts = 0
    
    with open(mcap_path, "rb") as f:
        reader = StreamReader(f)
        for record in reader.records:
            if hasattr(record, "log_time"):
                min_ts = min(min_ts, record.log_time)
                max_ts = max(max_ts, record.log_time)
    
    return min_ts, max_ts
