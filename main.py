import cv2
import torch
from transformers import pipeline
from collections import defaultdict

# Load a lightweight object detection model from Hugging Face
model_name = "hustvl/yolos-tiny"  # Tiny Vision Transformer for Object Detection
detector = pipeline("object-detection", model=model_name, framework="pt")

# Initialize video capture (replace 0 with video path)
cap = cv2.VideoCapture(0)
vehicle_classes = ["car", "motorcycle", "bus", "truck"]
person_class = "person"

# Store vehicle-person associations
vehicle_occupants = defaultdict(list)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Perform detection
    results = detector(frame)
    
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
        px, py = person["xmin"] + person["width"]/2, person["ymin"] + person["height"]/2
        
        for vehicle in current_vehicles:
            vx = vehicle["box"]["xmin"]
            vy = vehicle["box"]["ymin"]
            vw = vehicle["box"]["width"]
            vh = vehicle["box"]["height"]
            
            # Check if person center is within vehicle bounds
            if (vx < px < vx+vw) and (vy < py < vy+vh):
                vehicle["occupants"].append(person)

    # Check for violations
    for vehicle in current_vehicles:
        occupant_count = len(vehicle["occupants"])
        
        # Draw vehicle box
        cv2.rectangle(frame, 
            (vehicle["box"]["xmin"], vehicle["box"]["ymin"]),
            (vehicle["box"]["xmin"]+vehicle["box"]["width"], 
             vehicle["box"]["ymin"]+vehicle["box"]["height"]),
            (0, 255, 0), 2)

        # Check for triple riding (motorcycle with >2 people)
        if vehicle["type"] == "motorcycle" and occupant_count > 2:
            cv2.putText(frame, "TRIPLE RIDING VIOLATION!", 
                        (vehicle["box"]["xmin"], vehicle["box"]["ymin"]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            # Trigger alert here (e.g., send to central dashboard)

    # Display frame
    cv2.imshow("Traffic Monitoring", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
