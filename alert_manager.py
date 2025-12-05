"""
Aircraft alert manager - triggers warnings for aircraft on final approach or high-confidence detections.

Outputs to:
1. Foxglove topic (JSON alert state)
2. GPIO pin on Jetson (or debug print on other platforms)
"""

import logging
import time
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from shapely.geometry import Point, Polygon

logger = logging.getLogger(__name__)

# Try to import GPIO, fall back to debug mode if not available
try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except (ImportError, RuntimeError):
    GPIO_AVAILABLE = False
    logger.info("RPi.GPIO not available - using debug print mode")


@dataclass
class DetectionEvent:
    """A YOLO detection event with timing."""
    camera_id: str
    confidence: float
    timestamp: float  # seconds since epoch


class AlertManager:
    """Manages aircraft alerts from ADS-B and YOLO detections."""
    
    def __init__(
        self,
        final_approach_polygon: List[Tuple[float, float]],
        yolo_confidence_threshold: float = 0.5,
        yolo_duration_threshold: float = 0.5,
        debounce_duration: float = 2.0,
        gpio_pin: int = 30,
        foxglove_context=None,
        alert_topic: str = "/aircraft/alert"
    ):
        """
        Args:
            final_approach_polygon: List of (lat, lon) tuples defining the final approach area
            yolo_confidence_threshold: Min confidence to trigger alert (default 0.5)
            yolo_duration_threshold: Min duration in seconds for sustained detection (default 0.5s)
            debounce_duration: Cooldown period after alert triggered (default 2.0s)
            gpio_pin: GPIO pin number (BOARD mode) for output
            foxglove_context: Foxglove context for publishing
            alert_topic: Foxglove topic name for alert state
        """
        self.final_approach_zone = Polygon(final_approach_polygon)
        self.yolo_conf_threshold = yolo_confidence_threshold
        self.yolo_duration_threshold = yolo_duration_threshold
        self.debounce_duration = debounce_duration
        self.gpio_pin = gpio_pin
        
        # Foxglove setup
        self.foxglove_context = foxglove_context
        self.alert_topic = alert_topic
        self._foxglove_channel = None
        
        # Alert state
        self.alert_active = False
        self.last_alert_time = 0.0
        
        # YOLO detection tracking
        self.recent_detections: List[DetectionEvent] = []
        
        # GPIO setup
        self._gpio_initialized = False
        if GPIO_AVAILABLE:
            try:
                GPIO.setmode(GPIO.BOARD)
                GPIO.setup(self.gpio_pin, GPIO.OUT, initial=GPIO.LOW)
                self._gpio_initialized = True
                logger.info(f"GPIO initialized on pin {self.gpio_pin}")
            except Exception as e:
                logger.warning(f"Failed to initialize GPIO: {e}")
                self._gpio_initialized = False
        
        # Initialize Foxglove channel
        if self.foxglove_context:
            self._init_foxglove_channel()
    
    def _init_foxglove_channel(self):
        """Initialize Foxglove channel for alert state."""
        try:
            from foxglove import Channel, Schema
            import json
            
            # Alert state schema
            schema_def = {
                "type": "object",
                "properties": {
                    "alert_active": {"type": "boolean"},
                    "adsb_triggered": {"type": "boolean"},
                    "yolo_triggered": {"type": "boolean"},
                    "timestamp": {"type": "number"},
                    "recent_detection_count": {"type": "integer"},
                }
            }
            
            self._foxglove_channel = Channel(
                topic=self.alert_topic,
                message_encoding="json",
                schema=Schema(
                    name="aircraft_alert",
                    encoding="jsonschema",
                    data=json.dumps(schema_def).encode("utf-8"),
                ),
                context=self.foxglove_context,
            )
            logger.info(f"Foxglove alert channel initialized: {self.alert_topic}")
        except Exception as e:
            logger.error(f"Failed to initialize Foxglove alert channel: {e}")
    
    def check_adsb_final_approach(self, aircraft_states: Dict) -> bool:
        """
        Check if any aircraft is in the final approach zone.
        
        Args:
            aircraft_states: Dict of ICAO -> AircraftState
            
        Returns:
            True if any aircraft is in final approach zone
        """
        for icao, state in aircraft_states.items():
            if state.lat is not None and state.lon is not None:
                point = Point(state.lon, state.lat)  # Note: shapely uses (lon, lat)
                if self.final_approach_zone.contains(point):
                    logger.info(f"Aircraft {icao} in final approach zone at ({state.lat:.6f}, {state.lon:.6f})")
                    return True
        return False
    
    def add_yolo_detection(self, camera_id: str, confidence: float, timestamp: float):
        """
        Add a YOLO detection event.
        
        Args:
            camera_id: Camera identifier
            confidence: Detection confidence (0-1)
            timestamp: Detection timestamp in seconds
        """
        if confidence >= self.yolo_conf_threshold:
            event = DetectionEvent(camera_id, confidence, timestamp)
            self.recent_detections.append(event)
            
            # Clean up old detections (keep last 5 seconds worth)
            cutoff_time = timestamp - 5.0
            self.recent_detections = [d for d in self.recent_detections if d.timestamp >= cutoff_time]
    
    def check_yolo_sustained_detection(self, current_time: float) -> bool:
        """
        Check if there's a sustained high-confidence YOLO detection.
        
        Returns:
            True if detections exceed duration threshold
        """
        if not self.recent_detections:
            return False
        
        # Find the earliest and latest high-confidence detection
        earliest = min(d.timestamp for d in self.recent_detections)
        latest = max(d.timestamp for d in self.recent_detections)
        
        detection_span = latest - earliest
        
        # Check if we have sustained detections over the threshold duration
        if detection_span >= self.yolo_duration_threshold:
            return True
        
        return False
    
    def update_alert_state(
        self,
        aircraft_states: Dict,
        yolo_detections: List[Tuple[str, float]],  # List of (camera_id, confidence)
        current_time: float
    ) -> bool:
        """
        Update alert state based on ADS-B and YOLO data.
        
        Args:
            aircraft_states: Dict of ICAO -> AircraftState
            yolo_detections: List of (camera_id, confidence) tuples from current frame
            current_time: Current timestamp in seconds
            
        Returns:
            True if alert should be active
        """
        # Check if we're in debounce period
        time_since_last_alert = current_time - self.last_alert_time
        if self.alert_active and time_since_last_alert < self.debounce_duration:
            # Still in debounce period, keep alert active
            return True
        
        # Add new YOLO detections
        for camera_id, confidence in yolo_detections:
            self.add_yolo_detection(camera_id, confidence, current_time)
        
        # Check both conditions
        adsb_triggered = self.check_adsb_final_approach(aircraft_states)
        yolo_triggered = self.check_yolo_sustained_detection(current_time)
        
        should_alert = adsb_triggered or yolo_triggered
        
        # Update state
        if should_alert and not self.alert_active:
            # Alert is newly triggered
            logger.warning(f"ALERT ACTIVATED - ADS-B: {adsb_triggered}, YOLO: {yolo_triggered}")
            self.alert_active = True
            self.last_alert_time = current_time
            self._set_output(True)
        elif not should_alert and self.alert_active:
            # Alert should be cleared (debounce period has passed)
            if time_since_last_alert >= self.debounce_duration:
                logger.info("Alert cleared after debounce period")
                self.alert_active = False
                self._set_output(False)
        
        # Publish to Foxglove
        self._publish_foxglove_alert(adsb_triggered, yolo_triggered, current_time)
        
        return self.alert_active
    
    def _set_output(self, state: bool):
        """
        Set the output state (GPIO or debug print).
        
        Args:
            state: True for alert on, False for alert off
        """
        if self._gpio_initialized:
            try:
                GPIO.output(self.gpio_pin, GPIO.HIGH if state else GPIO.LOW)
                logger.info(f"GPIO pin {self.gpio_pin} set to {'HIGH' if state else 'LOW'}")
            except Exception as e:
                logger.error(f"Failed to set GPIO output: {e}")
        else:
            # Debug print mode
            print(f"[GPIO DEBUG] Outputting {'HIGH (ALERT)' if state else 'LOW'} to pin {self.gpio_pin}")
    
    def _publish_foxglove_alert(self, adsb_triggered: bool, yolo_triggered: bool, timestamp: float):
        """Publish alert state to Foxglove."""
        if not self._foxglove_channel:
            return
        
        try:
            alert_data = {
                "alert_active": self.alert_active,
                "adsb_triggered": adsb_triggered,
                "yolo_triggered": yolo_triggered,
                "timestamp": timestamp,
                "recent_detection_count": len(self.recent_detections)
            }
            
            self._foxglove_channel.log(alert_data)
        except Exception as e:
            logger.error(f"Failed to publish alert to Foxglove: {e}")
    
    def cleanup(self):
        """Clean up resources."""
        if self._gpio_initialized:
            try:
                GPIO.cleanup()
                logger.info("GPIO cleaned up")
            except Exception as e:
                logger.error(f"Failed to cleanup GPIO: {e}")
