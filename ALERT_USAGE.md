# Alert System Usage Guide

## Overview

The alert system monitors both ADS-B aircraft positions and YOLO detections to trigger warnings when aircraft are on final approach or when high-confidence visual detections occur.

## Features

1. **ADS-B Final Approach Detection**: Triggers when aircraft enters defined polygon
2. **YOLO Sustained Detection**: Triggers on >0.5 confidence for >0.5 seconds
3. **Debouncing**: 2-second cooldown after alert to handle camera transitions
4. **Dual Output**:
   - Foxglove JSON topic (`/aircraft/alert`) with alert state
   - GPIO pin (pin 30, BOARD mode) or debug print for LED control

## Command Line Arguments

```bash
--enable-alerts              # Enable the alert system
--alert-polygon "lat1,lon1;lat2,lon2;lat3,lon3;..."  # Final approach polygon
--alert-yolo-conf 0.5        # YOLO confidence threshold (default: 0.5)
--alert-yolo-duration 0.5    # YOLO sustained duration in seconds (default: 0.5)
--alert-debounce 2.0         # Debounce duration in seconds (default: 2.0)
--alert-topic /aircraft/alert  # Foxglove topic name
--alert-gpio-pin 30          # GPIO pin number (BOARD mode)
```

## Example Usage

### Define Your Final Approach Polygon

For example, for runway 33 at KBED (Bedford, MA), you might define:

```bash
POLYGON="42.471,-71.290;42.471,-71.280;42.450,-71.280;42.450,-71.290"
```

This creates a rectangular zone extending from the runway threshold northward along the approach path.

### Run with Alerts Enabled

```bash
python ./run_model_live.py \
  --source mcap \
  --mcap ~/Downloads/merged_spliced.mcap \
  --topics /camera/0/image/compressed /camera/1/image/compressed \
  --start-offset 3000 \
  --preroll 2.0 \
  --no-realtime \
  --class-filter airplane \
  --conf 0.10 \
  --foxglove-host 0.0.0.0 \
  --foxglove-port 8765 \
  --adsb-mcap ~/Downloads/adsb-decodes/20251025-103624-adsb-decode.mcap \
  --adsb-geojson-topic /aircraft/positions \
  --enable-alerts \
  --alert-polygon "42.471,-71.290;42.471,-71.280;42.450,-71.280;42.450,-71.290" \
  --alert-yolo-conf 0.5 \
  --alert-yolo-duration 0.5 \
  --alert-debounce 2.0 \
  --no-display
```

## Alert Logic

### Trigger Conditions (OR logic)

Alert activates when **EITHER**:

1. **ADS-B Trigger**: Any aircraft with valid position is inside the final approach polygon
2. **YOLO Trigger**: High-confidence detections (>= threshold) sustained for >= duration threshold

### Debouncing

Once alert triggers, it remains active for the debounce duration (default 2 seconds), even if conditions clear. This handles:
- Aircraft transitioning between cameras
- Temporary loss of YOLO detection
- Brief ADS-B data gaps

## Foxglove Output

The alert state publishes to Foxglove as JSON on the configured topic (default: `/aircraft/alert`):

```json
{
  "alert_active": true,
  "adsb_triggered": true,
  "yolo_triggered": false,
  "timestamp": 1761406425.5,
  "recent_detection_count": 3
}
```

## GPIO Output (Jetson)

On Jetson with RPi.GPIO available:
- **HIGH (3.3V)**: Alert active
- **LOW (0V)**: No alert

Pin configuration uses BOARD mode (physical pin numbering).

### Sample LED Circuit

```
GPIO Pin 30 ----[220Ω resistor]----[LED]----GND
```

On non-Jetson systems (laptop), GPIO commands print debug output instead:
```
[GPIO DEBUG] Outputting HIGH (ALERT) to pin 30
[GPIO DEBUG] Outputting LOW to pin 30
```

## Testing

Run the test script to verify alert logic without real data:

```bash
python test_alerts.py
```

This tests:
1. Aircraft outside polygon (no alert)
2. Aircraft inside polygon (alert triggers)
3. Debounce behavior
4. Sustained YOLO detections
5. Low-confidence YOLO (no alert)

## Monitoring in Foxglove

1. Open Foxglove Studio
2. Connect to `ws://localhost:8765`
3. Add a "JSON" panel
4. Subscribe to `/aircraft/alert`
5. Watch for `alert_active: true` when conditions trigger

You can also visualize the final approach polygon by adding a "Map" panel and subscribing to `/aircraft/positions` (GeoJSON) to see aircraft positions relative to the runway.
