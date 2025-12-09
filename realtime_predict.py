import torch
import cv2
import numpy as np

# Load the trained YOLOv5 model (use `weights/best.pt` in this repo)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='weights/best.pt', force_reload=True)
model.conf = 0.05  # Confidence threshold
model.iou = 0.3    # NMS IoU threshold

# Use the phone camera (replace with the index you found)
device_index = 1  # Replace with your phone camera index (e.g., 1)
cap = cv2.VideoCapture(device_index)
if not cap.isOpened():
    print("Error: Could not open camera")
    exit()

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")
        break

    # Preprocess frame (resize and adjust brightness/contrast)
    frame = cv2.resize(frame, (640, 640))  # Match the training image size (640x640)
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)  # Increase brightness/contrast

    # Perform inference
    results = model(frame)

    # Get detection results
    detections = results.pandas().xyxy[0]
    print("Detections:", detections)

    # Manually render bounding boxes with confidence scores
    for _, detection in detections.iterrows():
        xmin, ymin, xmax, ymax = int(detection['xmin']), int(detection['ymin']), int(detection['xmax']), int(detection['ymax'])
        confidence = detection['confidence']
        label_name = detection['name']
        
        # Adjust confidence threshold for students_not_cheating to balance detections
        if label_name == 'students_not_cheating' and confidence < 0.1:
            continue  # Skip low-confidence not_cheating detections
        if label_name == 'students_cheating' and confidence < 0.05:
            continue  # Skip low-confidence cheating detections

        label = f"{label_name} {confidence:.2f}"
        color = (0, 255, 0) if label_name == 'students_not_cheating' else (0, 0, 255)  # Green for not_cheating, red for cheating
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the frame in the "Cheating Detection" window
    cv2.imshow("Cheating Detection", frame)

    # Break on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()