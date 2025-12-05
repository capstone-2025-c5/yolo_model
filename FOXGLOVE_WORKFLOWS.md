# Foxglove Real-Time Streaming Workflows

This guide provides step-by-step workflows for streaming YOLO detections from Jetson to your laptop in real-time using Foxglove.

## Prerequisites

- **Jetson Device**: Running YOLO inference script
- **Laptop**: Running Foxglove app (https://app.foxglove.dev)
- **Network**: Both devices on same network (or with port forwarding)
- **Dependencies**: 
  ```bash
  pip install foxglove-sdk mcap opencv-python ultralytics av numpy
  ```

## Workflow 1: Quick Start (Local Network)

### On Jetson:
```bash
# Terminal 1: Start YOLO inference with Foxglove streaming
cd ~/path/to/yolo_model
python ./run_model_live.py \
  --source mcap \
  --mcap ~/Downloads/merged_spliced.mcap \
  --topics /camera/0/image/compressed /camera/1/image/compressed \
  --class-filter airplane \
  --conf 0.10 \
  --foxglove-host 0.0.0.0 \
  --foxglove-port 8765 \
  --no-display
```

### On Laptop:
1. Open https://app.foxglove.dev
2. Click **Open connection** → **WebSocket**
3. Enter: `ws://<jetson-ip>:8765` (e.g., `ws://192.168.1.100:8765`)
4. Click **Open**
5. In the layout panel:
   - Add **Image** panel
   - Subscribe to `/camera/0/image/compressed/detections`
   - Add another **Image** panel
   - Subscribe to `/camera/1/image/compressed/detections`

**Result**: Real-time annotated video from both cameras!

## Workflow 2: Save While Streaming

Stream to laptop AND save results to file:

### On Jetson:
```bash
python ./run_model_live.py \
  --source mcap \
  --mcap ~/Downloads/merged_spliced.mcap \
  --topics /camera/0/image/compressed /camera/1/image/compressed \
  --class-filter airplane \
  --conf 0.10 \
  --foxglove-host 0.0.0.0 \
  --foxglove-port 8765 \
  --out-mcap ~/results/detections_output.mcap \
  --no-display
```

**Benefits**:
- View live on Foxglove in real-time
- File saved for offline analysis later
- Can replay in Foxglove: File → Open file → select `.mcap`

## Workflow 3: Detection Metadata & Protobuf

Get both video AND detailed detection data:

### On Jetson:
```bash
python ./run_model_live.py \
  --source mcap \
  --mcap ~/Downloads/merged_spliced.mcap \
  --topics /camera/0/image/compressed /camera/1/image/compressed \
  --class-filter airplane \
  --conf 0.10 \
  --foxglove-host 0.0.0.0 \
  --foxglove-port 8765 \
  --out-pb ~/results/plane_states.pb \
  --no-display
```

### On Laptop (Foxglove):
1. Connect to WebSocket as above
2. Add **Protobuf** or **JSON** panel
3. Subscribe to `/detect/plane_state`

**Available data**:
- `plane_present`: Boolean detection status
- `plane.x, plane.y`: Normalized bbox center coordinates
- `plane.confidence`: Detection confidence (0-1)
- `plane.width, plane.height`: Normalized bbox dimensions
- `timestamp_unix_ns`: Frame timestamp
- `frame_seq`: Frame sequence number
- `source_topic`: Which camera topic

## Workflow 4: Performance Tuning

For optimal performance on Jetson:

```bash
# Reduced model size + lower confidence threshold
python ./run_model_live.py \
  --source mcap \
  --mcap ~/Downloads/merged_spliced.mcap \
  --topics /camera/0/image/compressed /camera/1/image/compressed \
  --weights yolov8n.pt \
  --imgsz 416 \
  --conf 0.05 \
  --foxglove-host 0.0.0.0 \
  --foxglove-port 8765 \
  --video-format jpeg \
  --no-realtime \
  --no-display
```

**Tuning parameters**:
- `--imgsz 416` (lower = faster, less accurate)
- `--weights yolov8n.pt` (nano model, fastest)
- `--video-format jpeg` (vs h264, lower CPU)
- `--no-realtime` (process as fast as possible)

## Workflow 5: Remote Access (Behind Firewall)

If Jetson is not directly accessible:

### Option A: SSH Tunneling
```bash
# On laptop: Forward local port to Jetson through SSH
ssh -L 8765:localhost:8765 jetson@jetson-ip

# Then connect to ws://localhost:8765
```

### Option B: ngrok (Public URL)
```bash
# On Jetson, install ngrok
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-arm64.zip
unzip ngrok-v3-stable-linux-arm64.zip

# Expose Foxglove port
./ngrok tcp 8765

# Copy the public URL from ngrok output and use in Foxglove
```

## Workflow 6: Multiple Jetson Devices

Stream from multiple Jetson boards to the same Foxglove instance:

### On Jetson 1:
```bash
python ./run_model_live.py \
  --source mcap \
  --mcap ~/data1.mcap \
  --topics /camera/0/image/compressed \
  --foxglove-host 0.0.0.0 \
  --foxglove-port 8765 \
  --no-display
```

### On Jetson 2:
```bash
python ./run_model_live.py \
  --source mcap \
  --mcap ~/data2.mcap \
  --topics /camera/0/image/compressed \
  --foxglove-host 0.0.0.0 \
  --foxglove-port 8766 \
  --no-display
```

### On Laptop:
- Open Foxglove 1: `ws://<jetson1-ip>:8765`
- Open Foxglove 2: `ws://<jetson2-ip>:8766`
- Use Foxglove's **Layouts** to organize multiple streams

## Monitoring & Debugging

### Check frame rate:
```bash
# Look for "FPS:" output with --show-fps flag
python ./run_model_live.py \
  ... \
  --show-fps
```

### Monitor Jetson resources:
```bash
# In another terminal on Jetson
watch -n 1 'nvidia-smi'  # GPU usage
top -p $(pgrep -f run_model_live.py)  # Process CPU/memory
```

### View streaming logs:
```bash
# Enable verbose logging
python ./run_model_live.py \
  ... \
  2>&1 | tee yolo_inference.log
```

### Test network connectivity:
```bash
# From laptop, test connection to Jetson
nc -zv <jetson-ip> 8765
# Should say: Connection successful
```

## Common Issues & Solutions

| Issue | Cause | Solution |
|-------|-------|----------|
| "Connection refused" | Port not open on Jetson | Check firewall: `sudo ufw allow 8765` |
| "Slow video playback" | Network bandwidth limited | Switch to JPEG: `--video-format jpeg` |
| "High CPU on Jetson" | YOLO model too large | Use nano model: `--weights yolov8n.pt --imgsz 416` |
| "Memory growing" | Memory leak in encoding | Restart script, use JPEG format |
| "Foxglove disconnects" | Network instability | Increase WebSocket timeout, use WiFi 5GHz |
| "No frames showing" | Topics not subscribed | Check topic names match exactly in Foxglove |

## Next Steps

- **Data Analysis**: Export detections to CSV using `tools/pb_to_csv.py`
- **Recording**: Use `--out-mcap` to record and replay later
- **Custom Visualization**: Create custom Foxglove layouts
- **Integration**: Use Foxglove API to build custom applications
