import cv2
from ultralytics import YOLO
import numpy as np
import os

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Create violation folder
os.makedirs("violations", exist_ok=True)

# Open video
cap = cv2.VideoCapture("videos/traffic.mp4")

# Stop line position
line_y = 350

frame_count = 0

# Traffic signal simulation
signal = "RED"

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    results = model(frame)

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):

            x1, y1, x2, y2 = map(int, box)

            label = model.names[int(cls)]

            if label in ["car", "truck", "bus", "motorbike"]:

                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(frame,label,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)

                center_y = int((y1 + y2) / 2)

                # Check violation
                if signal == "RED" and center_y > line_y:

                    cv2.putText(frame,"VIOLATION",(x1,y1-30),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

                    cv2.imwrite(f"violations/violation_{frame_count}.jpg",frame)

    # Draw stop line
    cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(255,0,0),3)

    # Show signal
    cv2.putText(frame,f"Signal: {signal}",(30,50),
                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)

    cv2.imshow("Traffic Violation Detection",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()