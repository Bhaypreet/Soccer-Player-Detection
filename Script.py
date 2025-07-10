import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Loading of model and using deep_sort

model = YOLO("best.pt")
tracker = DeepSort(max_age=30, n_init=3)

# Adjusting the rectangle boxes for tracking and detection

def xyxy_to_xywh(xyxy):
    x1, y1, x2, y2 = xyxy
    return [x1, y1, x2 - x1, y2 - y1]

def get_detections(frame, iou_thresh=0.5, conf_thresh=0.5):
    results = model(frame, iou=iou_thresh, conf=conf_thresh)
    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        label = model.names[cls_id]

        if label in ["player", "goalkeeper"]:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            xywh = xyxy_to_xywh(xyxy)
            detections.append((xywh, conf, label))

    return detections

def draw_tracks(frame, tracks):
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        x1, y1, x2, y2 = [int(coord) for coord in track.to_ltrb()]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
  
# Input / Output videos  

def main():
    cap = cv2.VideoCapture("soccer2.mp4")  # Loading the input video
    width = int(cap.get(3))
    height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("clean_output.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))  # Output video

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = get_detections(frame)
        tracks = tracker.update_tracks(detections, frame=frame)
        draw_tracks(frame, tracks)

        out.write(frame)
        cv2.imshow("Soccer Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
