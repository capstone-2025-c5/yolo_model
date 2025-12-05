# Multi-Camera YOLO Inference with Foxglove Streaming

This document explains how to run YOLO inference on multiple camera streams simultaneously and stream the annotated results to Foxglove for visualization on your laptop.

## Setup

1. Ensure you have all dependencies:
   ```bash
   pip install -r requirements.txt
   pip install foxglove-sdk
   ```

2. For H.264 video compression (optional), ensure PyAV is installed:
   ```bash
   pip install av
   ```

## Running Multi-Camera Processing

### Option 1: Stream to Foxglove Server (Live Visualization)

This is the recommended approach for real-time monitoring on your laptop:

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
  --no-display
```

Then on your laptop, open Foxglove and connect to `ws://<jetson-ip>:8765`

### Option 2: Write to MCAP File (Batch Processing)

For offline analysis or archival:

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
  --out-mcap output.mcap \
  --no-display
```

Then open `output.mcap` in Foxglove locally.

### Option 3: Both Streaming and File Output

Simultaneously stream to Foxglove and save to a file:

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
  --out-mcap output.mcap \
  --no-display
```

## Parameters Explained

### Input Parameters
- `--source mcap`: Read from MCAP file
- `--mcap <path>`: Path to MCAP file
- `--topics <topic1> <topic2> ...`: Multiple input topics to process in parallel
- `--start-offset 145`: Start 145 seconds into recording
- `--preroll 2.0`: Pre-load 2 seconds of data before start time (for keyframes)
- `--no-realtime`: Process as fast as possible (don't pace to real-time)

### YOLO Parameters
- `--class-filter airplane`: Only detect airplanes (or other classes)
- `--conf 0.10`: Confidence threshold (10%)
- `--weights yolov8n.pt`: Model weights (default)
- `--imgsz 640`: Input size for YOLO

### Output Parameters
- `--out-mcap output.mcap`: Write results to MCAP file
- `--out-pb detections.pb`: Optionally also write PlaneState protobuf stream
- `--topic-out /detect/plane_state`: Topic for detection metadata
- `--video-format jpeg`: Compression format for annotated video (`jpeg` or `h264`)
- `--no-display`: Don't show local display (recommended for Jetson)

### Foxglove Streaming Parameters
- `--foxglove-host 0.0.0.0`: Foxglove server listening address
  - Use `0.0.0.0` to accept connections from any IP (remote access)
  - Use `localhost` for local-only connections
- `--foxglove-port 8765`: WebSocket port for Foxglove connections (default: 8765)

## Understanding the Output

When you run with `--foxglove-host`, annotated video and detection data are streamed in real-time to Foxglove clients.

### Output Topics

For each input camera topic:
- **`/camera/0/image/compressed/detections`**: Annotated video frames with bounding boxes (JPEG/H.264)
- **`/camera/1/image/compressed/detections`**: Annotated video frames with bounding boxes (JPEG/H.264)
- **`/detect/plane_state`**: Protobuf messages containing detection metadata (coordinates, confidence, etc.)

## Viewing in Foxglove

### From Jetson (Streaming Mode):

1. On Jetson, run the script with `--foxglove-host 0.0.0.0`
2. On your laptop, open Foxglove (https://app.foxglove.dev)
3. Click **Open connection** → **WebSocket** → enter `ws://<jetson-ip>:8765`
4. Add layout panels:
   - **Image** panel → subscribe to `/camera/0/image/compressed/detections`
   - **Image** panel → subscribe to `/camera/1/image/compressed/detections`
   - **Protobuf** or **JSON** panel → subscribe to `/detect/plane_state` to view detection details

### From File (Batch Mode):

1. Run the script with `--out-mcap output.mcap`
2. Transfer the file to your laptop
3. Open Foxglove → **Open local file** → select `output.mcap`
4. Add the same layout panels as above

## Performance Notes

- **Processing Speed**: Multi-threaded processing means each camera stream runs on its own thread
- **JPEG vs H.264**: 
  - JPEG: Larger bandwidth (~1-3 Mbps per camera at 30 FPS), simpler encoding, universally supported
  - H.264: Lower bandwidth (~200-500 Kbps per camera at 30 FPS), more CPU-intensive encoding
- **Jetson**: For best performance on Jetson with Foxglove streaming, use JPEG format
- **Network**: Ensure sufficient bandwidth between Jetson and laptop (recommend gigabit Ethernet or 5GHz WiFi)

## Troubleshooting

### "Failed to connect to Foxglove server"
- Ensure `--foxglove-host` is set to a reachable IP (not `localhost` if connecting from remote)
- Check firewall rules on Jetson (port 8765 should be open)
- Verify network connectivity: `ping <jetson-ip>`

### "Foxglove server failed to start"
- Try a different port: `--foxglove-port 9000`
- Check if port is already in use: `lsof -i :8765`

### High CPU usage
- Reduce `--imgsz` (e.g., from 640 to 416)
- Switch to JPEG format: `--video-format jpeg`
- Reduce confidence threshold to skip post-processing on low-confidence detections
- Use `--no-realtime` to skip realtime pacing overhead

### Memory usage growing
- Ensure Foxglove video encoding is being flushed properly
- Use JPEG format instead of H.264
- Process fewer topics in parallel
- Monitor with `nvidia-smi` on Jetson

### Video appears frozen in Foxglove
- Ensure Jetson has consistent network connectivity
- Try increasing network buffer size if available
- Verify data is flowing: check for "[MCAP/Foxglove] Writing frame" messages in logs

## Single-Camera Mode (Original Behavior)

If you don't specify `--topics`, the script reverts to single-camera mode:

```bash
python ./run_model_live.py \
  --source mcap \
  --mcap ~/Downloads/merged_spliced.mcap \
  --topic /camera/0/image/compressed \
  --start-offset 145 \
  --out-mcap output.mcap \
  --no-display
```

This is backward-compatible with the original implementation.

## Example: Complete Jetson Setup

```bash
#!/bin/bash
# Run YOLO inference on Jetson, stream to Foxglove on laptop

# Get Jetson IP
JETSON_IP=$(hostname -I | awk '{print $1}')
echo "Jetson IP: $JETSON_IP"

# Run inference with Foxglove streaming
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
  --video-format jpeg \
  --no-display

echo "Connect to: ws://$JETSON_IP:8765"
```

