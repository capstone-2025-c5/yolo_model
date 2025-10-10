# General capture defaults
TARGET_TITLE_CONTAINS = "X-Plane"  # idk if x-plane would have different window titles but this needs to match
TARGET_FPS = 30
TARGET_W = 1280
TARGET_H = 720

# Choose source at runtime via CLI flag, but set defaults here too
DEFAULT_SOURCE = "gst_window"  # options: "gst_window", "camera"

# YOLO defaults
YOLO_WEIGHTS = "yolov8n.pt"
YOLO_CONF = 0.25
YOLO_IMGSZ = 640
