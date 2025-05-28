from ultralytics import YOLO
import cv2

# Load YOLOv8 helmet detection model from Hugging Face
model = YOLO('keremberke/yolov8m-helmet-detection')

# Video input (or replace with 0 for webcam)
video_path = 'test_video.mp4'
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)
    frame_with_boxes = results[0].plot()

    # Violation detection logic (example: no helmet or 3 people on one bike)
    for result in results:
        boxes = result.boxes
        classes = result.names
        
        helmet_count = sum(1 for c in boxes.cls if classes[int(c)] == 'helmet')
        person_count = sum(1 for c in boxes.cls if classes[int(c)] == 'person')
        
        if person_count >= 3:
            print("Triple riding detected!")
        if helmet_count < person_count:
            print("Helmet violation detected!")

    cv2.imshow("Violations", frame_with_boxes)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
