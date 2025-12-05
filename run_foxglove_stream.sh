#!/bin/bash
# Example: Run YOLO inference on Jetson with Foxglove streaming to laptop

# Configuration
MCAP_FILE="${HOME}/Downloads/merged_spliced.mcap"
JETSON_PORT=8765
CONFIDENCE=0.10
START_OFFSET=145
PREROLL=2.0

# Verify MCAP file exists
if [ ! -f "$MCAP_FILE" ]; then
    echo "Error: MCAP file not found at $MCAP_FILE"
    exit 1
fi

# Get local IP address for connection info
LOCAL_IP=$(hostname -I | awk '{print $1}')
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP="localhost"
fi

echo "=========================================="
echo "YOLO Inference with Foxglove Streaming"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  MCAP File: $MCAP_FILE"
echo "  Confidence Threshold: $CONFIDENCE"
echo "  Start Offset: ${START_OFFSET}s"
echo "  Preroll: ${PREROLL}s"
echo ""
echo "Output:"
echo "  Foxglove Server: ws://$LOCAL_IP:$JETSON_PORT"
echo ""
echo "On your laptop:"
echo "  1. Open https://app.foxglove.dev"
echo "  2. Click 'Open connection' -> 'WebSocket'"
echo "  3. Enter: ws://$LOCAL_IP:$JETSON_PORT"
echo "  4. Add Image panels for:"
echo "     - /camera/0/image/compressed/detections"
echo "     - /camera/1/image/compressed/detections"
echo ""
echo "Starting inference..."
echo "=========================================="
echo ""

python ./run_model_live.py \
  --source mcap \
  --mcap "$MCAP_FILE" \
  --topics /camera/0/image/compressed /camera/1/image/compressed \
  --start-offset $START_OFFSET \
  --preroll $PREROLL \
  --no-realtime \
  --class-filter airplane \
  --conf $CONFIDENCE \
  --foxglove-host 0.0.0.0 \
  --foxglove-port $JETSON_PORT \
  --video-format jpeg \
  --no-display

echo ""
echo "Inference complete."
