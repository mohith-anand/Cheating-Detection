# Real-Time Exam Cheating Detection
**Fine-tuned YOLOv5s • Live webcam / phone camera • ~200-image public dataset**

## Project Overview
Real-time proctoring system that detects cheating vs non-cheating behavior using only a webcam or phone camera.

- Fine-tuned official **Ultralytics YOLOv5s** via transfer learning  
- ~200-image public dataset  
- Classes: `students_cheating` (red) • `students_not_cheating` (green)  
- Runs **25–40 FPS on normal laptop CPU** — no GPU needed  

## Step-by-Step Setup

### 1. Download the dataset
1. Go to https://universe.roboflow.com/kattal/exam-cheating  
2. **Use ExamDataSet v1** → **YOLOv5** format → Download ZIP  
3. Extract and **rename the folder to `dataset`**  
4. Final structure must be:

```
your-project-folder/
└── dataset/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/                (can be empty)
```

### 2. Clone this repository
```bash
git clone https://github.com/mohith-anand/Cheating-Detection.git
cd exam-cheating-detection
```

### 3. Create virtual environment and install dependencies
```bash
python -m venv venv
# Windows → venv\Scripts\activate
# Mac/Linux → source venv/bin/activate

pip install -r requirements.txt
```

### 4. Run real-time detection
```bash
python realtime_predict.py
```

Press **q** to quit • Phone camera → index 1 (DroidCam / IP Webcam)

## Option: Train Your Own Model 
Want to create your own `best.pt` instead of using mine?

**Important disclaimer**:  
My project does **not** contain the full YOLOv5 source code.  
To perform transfer learning yourself, you must use the **official Ultralytics YOLOv5 repository**.

```bash
# 1. Clone the official YOLOv5 repo (required for training)
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt

# 2. Train (from the yolov5 folder)
python train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data ../data.yaml \
  --weights yolov5s.pt \
  --patience 30 \
  --cache ram \
  --project ../runs

# Best model will be saved at: ../runs/exp/weights/best.pt
# Copy it back to the main project folder and rename to best.pt
```

After that, replace my `best.pt` with your new one.

## Final Folder Structure
```
exam-cheating-detection/
├── dataset/              ← you add from ZIP
├── data.yaml
├── best.pt               ← my model (or your new one)
├── realtime_predict.py
├── requirements.txt
├── train.py             
└── README.md
```

## Credits
- **Dataset**: Exam Cheating Dataset (v1) by **KAttal** on Roboflow  
  https://universe.roboflow.com/kattal/exam-cheating  
- **Model base**: Official **Ultralytics YOLOv5**  
  https://github.com/ultralytics/yolov5  

Thanks to KAttal for sharing the dataset openly!