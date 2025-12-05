# Foxglove Visualization Guide

## Running the Full Pipeline

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

The server will start and listen on `0.0.0.0:8765` (accessible from any machine on the network).

## Connecting in Foxglove Viewer

1. Open Foxglove Viewer on your laptop
2. Click "Open connection"
3. Enter WebSocket URL: `ws://<jetson-ip>:8765`
   - If on same machine: `ws://localhost:8765`
   - If on network: `ws://192.168.x.x:8765` (use Jetson's IP)
4. Click "Open"

## Available Topics

### Video Streams (RawImage)
- **`/camera/0/image/compressed/detections`** - Annotated video from front camera
- **`/camera/1/image/compressed/detections`** - Annotated video from rear camera

**Visualization:**
- Add "Image" panel
- Select either camera topic
- Displays video with bounding boxes and confidence labels

### Confidence Scores (JSON)
- **`/camera/0/image/compressed/confidence`** - YOLO confidence from camera 0
- **`/camera/1/image/compressed/confidence`** - YOLO confidence from camera 1

**Visualization:**
- Add "Scalar" or "Plot" panel
- Select topic to see confidence trends over time

### Aircraft Positions (GeoJSON)
- **`/aircraft/positions`** - Real-time ADS-B aircraft locations on map

**Visualization:**
- Add "Map" panel
- Select `/aircraft/positions` topic
- **Features:**
  - Green points show aircraft positions
  - Labels show flight callsigns (or ICAO code)
  - Hover/click to see: altitude, track, speed, ICAO code
  - Real-time updates as data streams

## Recommended Layout

For optimal viewing:

1. **Top Left**: Map panel with `/aircraft/positions`
   - Shows real-time aircraft positions
   - Provides geographic context

2. **Top Right**: Image panel with `/camera/0/image/compressed/detections`
   - Main video feed
   - Shows YOLO detections

3. **Bottom Right**: Image panel with `/camera/1/image/compressed/detections`
   - Alternate camera view
   - Dual perspective on detections

4. **Bottom Left**: Plot panel with `/camera/0/image/compressed/confidence`
   - Shows confidence confidence over time
   - Helps identify detection strength

## GeoJSON Styling

Aircraft markers use this styling:
- **Color**: Green (`#00ff00`)
- **Opacity**: 0.8
- **Weight**: 2 pixels
- **Name field**: Flight callsign (if available) or "Aircraft [ICAO]"

Properties available on hover:
- `name`: Callsign or aircraft identifier
- `icao`: 6-digit ICAO code
- `callsign`: Flight number
- `altitude_m`: Current altitude in meters
- `track_deg`: Aircraft heading (0-360°)
- `speed_knots`: Ground speed in knots
- `timestamp`: Position timestamp

## Performance Tips

1. **Frame Rate**: Runs at ~30 FPS (limited by MCAP playback, not Foxglove)
2. **Network**: Works well on local network; WebSocket efficient even for remote viewing
3. **CPU**: Single-threaded inference on Jetson; bottleneck is YOLO processing
4. **Memory**: ~2GB for full pipeline (Foxglove + YOLO + ADS-B)

## Troubleshooting

### "Connection refused"
- Ensure `--foxglove-host 0.0.0.0` is set
- Check firewall allows port 8765
- Verify correct IP address for remote connections

### Aircraft markers not appearing
- Confirm `/aircraft/positions` topic is listed in viewer
- Check ADS-B MCAP file path is correct
- Verify time sync shows overlap between datasets

### Video not appearing
- Ensure camera topics appear in Foxglove topic list
- Check MCAP file path is valid
- Verify `--start-offset` and `--preroll` values

### Viewer shows "No data"
- Wait 5-10 seconds for pipeline initialization
- Check console output for errors
- Verify Foxglove client connected: `"Registered client"` in logs

## Network Connectivity

### On Same Machine
```bash
# Server (Jetson or local)
ws://localhost:8765

# Viewer (same machine)
Connect to: ws://localhost:8765
```

### Over Network
```bash
# Get Jetson IP
hostname -I  # Linux/Mac
ipconfig     # Windows

# Server (Jetson on network)
ws://<jetson-ip>:8765

# Viewer (laptop on same network)
Connect to: ws://<jetson-ip>:8765
```

### SSH Tunneling (if needed)
```bash
# On your laptop, forward remote port
ssh -L 8765:localhost:8765 user@jetson

# Then connect to
ws://localhost:8765
```
