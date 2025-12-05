#!/usr/bin/env python3
"""Test the alert manager with sample data."""

import sys
import time
from dataclasses import dataclass
from typing import Optional

# Mock AircraftState for testing
@dataclass
class AircraftState:
    icao: str
    callsign: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    altitude_m: Optional[float]
    track_degrees: Optional[float]
    ground_speed_knots: Optional[float]
    timestamp: float

from alert_manager import AlertManager

# Define a test polygon around a runway final approach (example coordinates)
# This is a rectangle approximating final approach to runway 33 at KBED
final_approach_polygon = [
  (-71.52218614751439, 42.45670821715322),
  (-71.53132961889035, 42.44207606904803),
  (-71.52355187632614, 42.43966519519566),
  (-71.51921708994236, 42.45609491852423),
  (-71.51963283876673, 42.45854816350533),
  (-71.52218614751439, 42.45670821715322),
]

print("=" * 70)
print("ALERT MANAGER TEST")
print("=" * 70)
print(f"\nFinal approach polygon: {len(final_approach_polygon)} points")
print("YOLO confidence threshold: 0.5")
print("YOLO duration threshold: 0.5s")
print("Debounce duration: 2.0s")
print("\nGPIO DEBUG MODE (RPi.GPIO not available)\n")

# Create alert manager without Foxglove context
alert_mgr = AlertManager(
    final_approach_polygon=final_approach_polygon,
    yolo_confidence_threshold=0.5,
    yolo_duration_threshold=0.5,
    debounce_duration=2.0,
    gpio_pin=30,
    foxglove_context=None
)

print("Test 1: Aircraft outside final approach zone")
print("-" * 70)
aircraft_outside = {
    "ABC123": AircraftState(
        icao="ABC123",
        callsign="N12345",
        lat=42.40,  # Too far south
        lon=-71.285,
        altitude_m=1000.0,
        track_degrees=330.0,
        ground_speed_knots=120.0,
        timestamp=time.time()
    )
}
alert_mgr.update_alert_state(aircraft_outside, [], time.time())
print(f"Alert active: {alert_mgr.alert_active}\n")

print("Test 2: Aircraft inside final approach zone (ADS-B trigger)")
print("-" * 70)
aircraft_inside = {
    "ABC123": AircraftState(
        icao="ABC123",
        callsign="N12345",
        lat=42.46,  # Inside polygon
        lon=-71.285,
        altitude_m=500.0,
        track_degrees=330.0,
        ground_speed_knots=85.0,
        timestamp=time.time()
    )
}
alert_mgr.update_alert_state(aircraft_inside, [], time.time())
print(f"Alert active: {alert_mgr.alert_active}\n")

# Wait for debounce
print("Test 3: Debounce period (2.5 seconds)")
print("-" * 70)
time.sleep(2.5)
alert_mgr.update_alert_state({}, [], time.time())
print(f"Alert active after debounce: {alert_mgr.alert_active}\n")

print("Test 4: High-confidence YOLO detections (sustained >0.5s)")
print("-" * 70)
current_time = time.time()
# Simulate detections over 0.6 seconds
for i in range(6):
    yolo_detections = [("camera_0", 0.85), ("camera_1", 0.75)]
    alert_mgr.update_alert_state({}, yolo_detections, current_time + i * 0.1)
print(f"Alert active after sustained YOLO: {alert_mgr.alert_active}\n")

print("Test 5: Low-confidence YOLO detections (should NOT trigger)")
print("-" * 70)
# Wait for debounce
time.sleep(2.5)
current_time = time.time()
for i in range(6):
    yolo_detections = [("camera_0", 0.3), ("camera_1", 0.2)]
    alert_mgr.update_alert_state({}, yolo_detections, current_time + i * 0.1)
print(f"Alert active with low confidence: {alert_mgr.alert_active}\n")

print("=" * 70)
print("✓ Alert manager test complete")
print("=" * 70)

# Cleanup
alert_mgr.cleanup()
