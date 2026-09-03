from fastapi import FastAPI, WebSocket, WebSocketDisconnect
# FastAPI = the main web framework
# WebSocket = represents a live, two-way connection with a browser
# WebSocketDisconnect = exception raised when the browser closes the connection

from ultralytics import YOLO  # the YOLOv8 object detection library
import cv2                    # OpenCV - handles video reading and image encoding
import asyncio                # needed for asyncio.sleep() to pace how fast we send frames
import base64                 # converts binary image data into text so it can go inside JSON

app = FastAPI()
# creates the FastAPI application instance - this is what uvicorn actually runs

model = YOLO("yolo11n.pt")
# loads the pretrained YOLOv8 "nano" model - reliable, well-tested COCO classes, no fine-tuning risk

VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# maps YOLO's internal COCO class ID numbers to readable vehicle names
# these specific numbers come from the COCO dataset YOLO was trained on

# --- Multi-lane video setup ---
LANE_VIDEOS = {
    "north": "traffic_video.mp4",
    "south": "traffic_video1.mp4",
    "east": "traffic_video2.mp4",
    "west": "WhatsApp Video 2026-08-21 at 10.15.06 PM.mp4",  # <-- confirm this matches your exact filename via `dir`
}
# one video file per lane, simulating 4 separate traffic camera feeds

lane_caps = {lane: cv2.VideoCapture(path) for lane, path in LANE_VIDEOS.items()}
# creates a separate cv2.VideoCapture object for EACH lane, used by /detect-all
# WebSocket connections use their OWN separate captures (see below) to avoid frame-stealing

# Verify every video actually opened successfully at startup - fail loudly and clearly, not silently later
for lane, capture in lane_caps.items():
    if not capture.isOpened():
        print(f"WARNING: Failed to open video for lane '{lane}': {LANE_VIDEOS[lane]}", flush=True)
        print(f"Check the filename matches exactly (spaces, capitalization, extension).", flush=True)


def count_vehicles_in_frame(frame):
    # shared helper function - takes one video frame, returns vehicle counts
    results = model(frame, classes=list(VEHICLE_CLASSES.keys()), conf=0.4, verbose=False)
    counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        counts[VEHICLE_CLASSES[cls_id]] += 1

    return counts


def detect_lane(lane):
    # reads ONE frame from a specific lane's video and returns its total vehicle count

    lane_cap = lane_caps[lane]
    ret, frame = lane_cap.read()

    if not ret:
        lane_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = lane_cap.read()

    if not ret or frame is None:
        # video is genuinely broken/unreadable even after reset - fail safely instead of crashing
        print(f"WARNING: Could not read a frame for lane '{lane}' - returning 0", flush=True)
        return 0

    counts = count_vehicles_in_frame(frame)
    return sum(counts.values())


@app.get("/detect-all")
def detect_all():
    result = {lane: detect_lane(lane) for lane in LANE_VIDEOS}
    print(f"Detect-All Route API - {result}", flush=True)
    return result


# --- Live WebSocket video + detection stream, one connection per lane ---

@app.websocket("/ws/{lane}")
async def websocket_lane(websocket: WebSocket, lane: str):
    if lane not in LANE_VIDEOS:
        await websocket.close()
        return

    await websocket.accept()

    lane_cap = cv2.VideoCapture(LANE_VIDEOS[lane])
    # fresh, private capture for THIS connection only - not shared with /detect-all

    if not lane_cap.isOpened():
        # video failed to open - close the connection cleanly instead of looping forever at 100% CPU
        print(f"ERROR: Could not open video for lane '{lane}' - closing WebSocket", flush=True)
        await websocket.close()
        lane_cap.release()
        return

    try:
        while True:
            ret, frame = lane_cap.read()
            if not ret:
                lane_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = lane_cap.read()
                if not ret or frame is None:
                    # still broken after reset - stop this connection instead of spinning forever
                    print(f"ERROR: Lane '{lane}' video unreadable - ending stream", flush=True)
                    break

            try:
                results = model(frame, classes=list(VEHICLE_CLASSES.keys()), conf=0.4, verbose=False)
                annotated = results[0].plot()

                counts = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}
                for box in results[0].boxes:
                    cls_id = int(box.cls[0])
                    counts[VEHICLE_CLASSES[cls_id]] += 1

                _, buffer = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frame_b64 = base64.b64encode(buffer).decode("utf-8")

                await websocket.send_json({
                    "lane": lane,
                    "frame": frame_b64,
                    "counts": counts,
                    "total": sum(counts.values())
                })

            except Exception as inner_error:
                # catches YOLO errors, encoding errors, or the client disconnecting mid-send
                # without this, a single bad frame or a mistimed disconnect could kill the loop
                # with an unhandled traceback instead of exiting cleanly
                print(f"Error processing frame for lane '{lane}': {inner_error}", flush=True)
                break

            await asyncio.sleep(0.1)
            # wait 100ms before the next frame - caps streaming at ~10 frames/second

    except WebSocketDisconnect:
        print(f"Client disconnected from {lane} feed", flush=True)

    finally:
        lane_cap.release()
        print(f"Released capture for {lane}", flush=True)