import cv2
from ultralytics import YOLO
from threading import Lock
import numpy as np

class YOLORunner:
    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.25, imgsz: int = 640, allowed_names=None):
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.class_names = self.model.names  # id->name
        self._inference_lock = Lock()  # Serialize inference calls
        
        # Map allowed class names -> ids (lowercase match)
        if allowed_names:
            allowed_lc = set(n.lower() for n in allowed_names)
            self.allowed_class_ids = [i for i, n in self.class_names.items() if n.lower() in allowed_lc]
        else:
            self.allowed_class_ids = None
        
        # Warm up the model with a dummy inference to trigger fusion
        # This must happen BEFORE any worker threads start
        self._warmup()

    def _warmup(self):
        """Warm up the model to ensure it's fully initialized and fused."""
        try:
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            _ = self.model.predict(
                source=dummy_frame,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
                classes=self.allowed_class_ids,
            )
            print("YOLO model warmup complete (model fused and ready)")
        except Exception as e:
            print(f"Warning: YOLO warmup failed: {e}")

    def infer(self, frame):
        """Thread-safe inference call."""
        with self._inference_lock:
            return self.model.predict(
                source=frame,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
                classes=self.allowed_class_ids  # <-- filter at inference if provided
            )

    def draw(self, frame, results):
        r = results[0]
        if getattr(r, "boxes", None) is None or len(r.boxes) == 0:
            return frame
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{self.class_names[cls]} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        return frame
