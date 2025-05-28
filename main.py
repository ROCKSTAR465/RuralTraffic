import cv2
import torch
from transformers import pipeline
from collections import defaultdict
from PIL import Image # Import the PIL library

# uncomment the below line to use in colab
# from google.colab.patches import cv2_imshow

# Load a lightweight object detection model from Hugging Face
model_name = "hustvl/yolos-tiny"  # Tiny Vision Transformer for Object Detection
detector = pipeline("object-detection", model=model_name, framework="pt")

# Initialize video capture (replace 0 with video path)
VIDEO_PATH = "path/to/your/test_file.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
vehicle_classes = ["car", "motorcycle", "bus", "truck"]
person_class = "person"

# Store vehicle-person associations
vehicle_occupants = defaultdict(list)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the OpenCV frame (BGR NumPy array) to a PIL Image (RGB)
    # Transformers pipeline expects RGB, OpenCV reads as BGR
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(frame_rgb)

    # Perform detection on the PIL image
    results = detector(pil_image)

    current_vehicles = []
    current_people = []

    # Process detections
    for result in results:
        label = result["label"]
        box = result["box"]

        if label in vehicle_classes:
            current_vehicles.append({
                "box": box,
                "type": label,
                "occupants": []
            })
        elif label == person_class:
            current_people.append(box)

    # Associate people with vehicles (simple proximity-based)
    for person in current_people:
        # Note: The box format from the detector for person is xmin, ymin, xmax, ymax
        px = (person["xmin"] + person["xmax"])/2
        py = (person["ymin"] + person["ymax"])/2

        for vehicle in current_vehicles:
            # Note: The box format for vehicle is also xmin, ymin, xmax, ymax
            vx_min = vehicle["box"]["xmin"]
            vy_min = vehicle["box"]["ymin"]
            vx_max = vehicle["box"]["xmax"]
            vy_max = vehicle["box"]["ymax"]

            # Check if person center is within vehicle bounds
            if (vx_min < px < vx_max) and (vy_min < py < vy_max):
                vehicle["occupants"].append(person)

    # Check for violations
    for vehicle in current_vehicles:
        occupant_count = len(vehicle["occupants"])

        # Draw vehicle box
        # Note: The box format used for drawing with cv2.rectangle should be (xmin, ymin), (xmax, ymax)
        cv2.rectangle(frame,
            (vehicle["box"]["xmin"], vehicle["box"]["ymin"]),
            (vehicle["box"]["xmax"], vehicle["box"]["ymax"]),
            (0, 255, 0), 2)

        # Check for triple riding (motorcycle with >2 people)
        if vehicle["type"] == "motorcycle" and occupant_count > 2:
            cv2.putText(frame, "TRIPLE RIDING VIOLATION!",
                        (vehicle["box"]["xmin"], vehicle["box"]["ymin"]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            # Trigger alert here (e.g., send to central dashboard)

    # Display frame
    # for using in colab uncomment the below line
    # cv2_imshow(frame)
    # for local system
    cv2.imshow("Traffic Monitoring", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
