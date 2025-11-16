import cv2
import numpy as np
import pygame
from ultralytics import YOLO
import argparse
import os

# -------------------------------
# 🚀 Initialize pygame for sound
# -------------------------------
pygame.mixer.init()
try:
    alert_sound = pygame.mixer.Sound("alert.wav")  # Make sure alert.wav exists
except Exception as e:
    print(f"[WARN] Audio file not found or failed to load: {e}")
    alert_sound = None

# -------------------------------
# 🎯 Argument parser
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="videos/road2.mp4", help="Path to video or 0 for webcam")
args = parser.parse_args()

# -------------------------------
# 🧠 Load YOLO model
# -------------------------------
print("[INFO] Loading YOLO model...")
model = YOLO("yolov8n.pt")  # lightweight and fast

# -------------------------------
# 🚗 Video capture
# -------------------------------
source = 0 if args.source == "0" else args.source
cap = cv2.VideoCapture(source)

if not cap.isOpened():
    print("[ERROR] Unable to open video source.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
print(f"[INFO] Processing... source={args.source} fps={fps}")

# -------------------------------
# 📏 Function: Calculate proximity
# -------------------------------
def get_proximity(frame, box):
    h, w, _ = frame.shape
    x1, y1, x2, y2 = map(int, box)
    box_width = x2 - x1
    frame_center = w // 2
    obj_center = (x1 + x2) // 2
    offset = abs(obj_center - frame_center)

    # proximity factor (0 = far, 1 = close)
    proximity = (box_width / w) * (1 - offset / w)
    return np.clip(proximity, 0, 1)

# -------------------------------
# 🎨 Main loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("[INFO] Video stream ended.")
        break

    # YOLO detections
    results = model(frame, stream=True)

    status_text = "🟢 SAFE DISTANCE"
    color = (0, 255, 0)
    max_proximity = 0

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            if model.names[cls] in ["car", "truck", "bus", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                proximity = get_proximity(frame, (x1, y1, x2, y2))
                max_proximity = max(max_proximity, proximity)

                # Color logic for individual boxes
                if proximity < 0.25:
                    box_color = (0, 255, 0)   # Green
                elif proximity < 0.45:
                    box_color = (0, 165, 255) # Orange
                else:
                    box_color = (0, 0, 255)   # Red

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame, model.names[cls], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2)

    # -------------------------------
    # 🚨 Global Alert Logic
    # -------------------------------
    if max_proximity < 0.25:
        status_text = "🟢 SAFE DISTANCE"
        color = (0, 255, 0)
    elif max_proximity < 0.45:
        status_text = "🟠 CAUTION: Maintain Distance"
        color = (0, 165, 255)
    else:
        status_text = "🔴 DANGER: COLLISION WARNING"
        color = (0, 0, 255)
        if alert_sound:
            try:
                alert_sound.play()
            except:
                pass

    # -------------------------------
    # 🖥️ Display status
    # -------------------------------
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 0.9, color, 2)

    cv2.imshow("🚘 ADAS Lite - Collision Detection", frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
