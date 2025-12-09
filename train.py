# train.py
# Simple & clean training script for your Exam Cheating Detection project
# Run with: python train.py

from ultralytics import YOLO

# Load YOLOv5s (best balance of speed & accuracy for small datasets)
# Automatically downloads the first time
model = YOLO("yolov5s.pt")          # you can also use yolov5n.pt, yolov5m.pt

# Train on your Roboflow dataset
results = model.train(
    data="data.yaml",       # your Roboflow data.yaml file
    epochs=100,             # will early-stop much earlier thanks to patience
    imgsz=640,
    batch=16,
    patience=30,            # stops training if no improvement for 30 epochs → perfect for 100 images
    cache="ram",            # speeds up training a lot
    project="runs",
    name="weights",
    exist_ok=True,
    pretrained=True
)

print("Training finished!")
print("Best model saved at: runs/weights/best.pt")