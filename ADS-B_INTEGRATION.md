# ADS-B + YOLO Multi-Camera Foxglove Integration

## Overview

Successfully integrated ADS-B aircraft tracking data with YOLO airplane detection from two camera streams, all streaming to Foxglove visualization in real-time.

## Architecture

### Components

1. **adsb_geojson_streamer.py** - ADS-B data processing
   - `ADSBDataReader`: Parses ADS-B MCAP files and extracts aircraft positions
   - `GeoJSONStreamer`: Streams aircraft positions as GeoJSON FeatureCollection to Foxglove
   - `time_sync_adsb_to_video()`: Synchronizes ADS-B and video timescales

2. **run_model_live.py** - Main inference pipeline (updated)
   - Multi-camera YOLO inference (two concurrent streams)
   - RawImage video streaming to Foxglove
   - Confidence score streaming
   - ADS-B GeoJSON streaming integrated into worker threads

### Data Flow

```
ADS-B MCAP                Video MCAP                  
    ↓                        ↓
[ADSBDataReader]      [MCAPFrameSource (x2)]
    ↓                        ↓
[Aircraft States]     [Frame Capture (x2)]
    ↓                        ↓
[GeoJSONStreamer]     [YOLO Inference (x2)]
    ↓                        ↓
[Foxglove Topics]    [Foxglove Topics]
    ↓                        ↓
/aircraft/positions    /camera/X/image/compressed/detections
                       /camera/X/image/compressed/confidence
```

## Foxglove Topics

### Video Detections (RawImage)
- `/camera/0/image/compressed/detections` - Annotated video stream from camera 0
- `/camera/1/image/compressed/detections` - Annotated video stream from camera 1
- Format: RawImage (RGB8, raw pixel data)
- Features: Bounding boxes, confidence labels

### Confidence Scores (JSON Scalar)
- `/camera/0/image/compressed/confidence` - YOLO confidence for camera 0
- `/camera/1/image/compressed/confidence` - YOLO confidence for camera 1
- Format: `{"timestamp": number, "value": 0-1, "label": "airplane"}`

### Aircraft Positions (GeoJSON)
- `/aircraft/positions` - Real-time ADS-B aircraft positions
- Format: GeoJSON FeatureCollection with Point geometry
- Properties: icao, callsign, altitude_m, track_deg, speed_knots

## Usage

### Full Pipeline Command

```bash
python ./run_model_live.py \
  --source mcap \
  --mcap ~/Downloads/merged_spliced.mcap \
  --topics /camera/0/image/compressed /camera/1/image/compressed \
  --start-offset 145 \
  --preroll 2.0 \
  --no-realtime \
  --class-filter airplane \
  --conf 0.10 \
  --foxglove-host 0.0.0.0 \
  --foxglove-port 8765 \
  --adsb-mcap ~/Downloads/adsb-decodes/20251025-103624-adsb-decode.mcap \
  --adsb-geojson-topic /aircraft/positions \
  --no-display
```

### Key Arguments
- `--adsb-mcap`: Path to ADS-B MCAP file
- `--adsb-geojson-topic`: Foxglove topic name for aircraft (default: `/aircraft/positions`)
- `--foxglove-host`: Foxglove server host (e.g., `0.0.0.0` for accessible from other machines)
- `--foxglove-port`: WebSocket port (default: 8765)

### Viewing in Foxglove

1. Open Foxglove viewer on another machine
2. Connect to WebSocket: `ws://<server-ip>:8765`
3. Topics available:
   - Select `/camera/0/image/compressed/detections` → RawImage layer
   - Select `/camera/1/image/compressed/detections` → RawImage layer (different window)
   - Select `/aircraft/positions` → GeoJSON layer with aircraft markers

## Implementation Details

### ADS-B Data Format

ADS-B MCAP files use foxglove.Log schema containing JSON arrays of aircraft:

```json
[
  {
    "hex": "a4c9f8",           // ICAO 24-bit code
    "flight": "DAL123",         // Flight identifier
    "lat": 42.1234,            // Latitude
    "lon": -71.5678,           // Longitude
    "alt_geom": 35000,         // Geometric altitude (m)
    "gs": 450,                 // Ground speed (knots)
    "track": 90.5              // Track (degrees)
  },
  ...
]
```

### Time Synchronization

- Detects temporal overlap between ADS-B and video datasets
- Applies video start offset (`--start-offset`) to align timestamps
- Logs overlap duration and any gaps
- Uses 60-second lookback window for aircraft matching

### Multi-Threading

Worker threads for each camera stream:
- Read frames from video MCAP
- Run YOLO inference (with serialized access via YOLORunner lock)
- Stream annotated video to Foxglove
- Stream confidence scores to Foxglove
- **NEW**: Retrieve current ADS-B aircraft and stream as GeoJSON
  - Uses frame timestamp to query aircraft positions
  - 60-second lookback tolerance for temporal alignment

## Testing

```bash
# Test ADS-B data loading and parsing
python test_adsb.py

# Show full pipeline command
python test_full_pipeline.py

# Quick 15-second integration test
timeout 20 python ./run_model_live.py [args]
```

## Files Modified

- `run_model_live.py`: Added ADS-B arguments, loading, and streaming integration
- `adsb_geojson_streamer.py`: NEW - Complete ADS-B processing pipeline
- `test_adsb.py`: NEW - Testing utility
- `test_full_pipeline.py`: NEW - Command reference

## Performance Considerations

- ADS-B parsing: ~2 seconds for 1944 messages (MCAP reading overhead)
- Aircraft lookup: O(n_messages) with 60-second window (fast with sorted list)
- GeoJSON streaming: JSON serialization once per frame
- RawImage streaming: Concurrent with video (minimal overhead)

## Future Enhancements

1. **Compound visualization**: Overlay aircraft flight paths on video
2. **Detection matching**: Correlate YOLO detections with ADS-B aircraft
3. **Filtering**: Show only aircraft types (regional jets, large transport, etc.)
4. **Telemetry**: Stream additional metrics (detection confidence vs ADS-B altitude, etc.)
5. **Real-time ADS-B**: Stream live ADS-B feed instead of MCAP file

## Verification

✅ ADS-B MCAP parsing working
✅ GeoJSON schema validated
✅ Time sync with video MCAP verified (1.9 hours overlap)
✅ Aircraft extraction confirmed (29 aircraft at test timestamp)
✅ Foxglove channel registration working
✅ Multi-camera streaming active
✅ Confidence scoring working
✅ Integration test passes (server starts, channels initialize, client connects)
