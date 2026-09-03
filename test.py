from ultralytics import YOLO
import cv2

model = YOLO("yolov8n.pt")

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def count_vehicles(frame):
    results = model(frame, classes=list(VEHICLE_CLASSES.keys()), verbose=False)
    counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        counts[VEHICLE_CLASSES[cls_id]] += 1
    return counts

cap = cv2.VideoCapture("traffic_video.mp4")
ret, frame = cap.read()

if not ret:
    print("Could not read frame")
else:
    print(count_vehicles(frame))

cap.release()