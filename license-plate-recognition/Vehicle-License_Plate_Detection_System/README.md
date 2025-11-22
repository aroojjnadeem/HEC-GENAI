#  Automatic Vehicle & License Plate Detection System

This project implements a **real-time vehicle detection, tracking, and license plate recognition system** using **YOLOv8**, **EasyOCR**, and **Kalman filter-based SORT tracking**.

The pipeline processes video input, detects vehicles and license plates, recognizes plate numbers, and generates an annotated video with bounding boxes and license plate information.

---

## 📌 Features

- 🚘 Vehicle detection (cars, trucks, buses, motorcycles) using **YOLOv8**
- 🔲 License plate detection using a **custom-trained YOLO model**
- 🎯 Vehicle tracking with **SORT (Kalman Filter + Hungarian Algorithm)**
- 🔡 License plate recognition with **EasyOCR**, including character correction
- 📈 Interpolation for smoother bounding box trajectories
- 🎥 Outputs an annotated video with vehicles, license plates, and recognized numbers

---

## 📂 Project Structure

```
LPR/
│
├── 📜 new.py               # Main pipeline script
├── 🚌 bus.py               # (Optional) Bus detection module
├── 📹 sample.mp4           # Example input video
├── 🤖 yolov8n.pt           # Pre-trained YOLOv8 vehicle model
├── 🔲 license_plate_detector.pt # Custom license plate detector model
└── 🎬 out.mp4              # Generated output video
```

---

## ⚙️ Installation

### 1️⃣ Create a Virtual Environment

It's highly recommended to use a virtual environment (Python 3.10 is suggested).

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 2️⃣ Install Dependencies

Install all required packages with pip.

```bash
pip install ultralytics opencv-python-headless numpy pandas scipy easyocr filterpy lap
```
> ⚠️ **Note:** If `lap` causes installation errors, you can skip it. The code will automatically fall back to `scipy.optimize`.

---

## ▶️ Usage

Run the main script from your terminal to start the process.

```bash
python new.py
```

The script will load `sample.mp4`, process it, and save the result as `out.mp4`.

---

## 📝 Notes & Configuration

- **Custom Video**: To use your own video, change the `video_path` variable in `new.py`.
- **Model Files**: Make sure `yolov8n.pt` and `license_plate_detector.pt` are in the same folder as the script.
- **Output Name**: The output file is named `out.mp4` by default but can be changed in the script.

---

## 🚀 Future Improvements

- 📡 **Real-time Stream Support**: Add functionality to process live RTSP streams.
- 🎯 **Enhanced OCR**: Improve accuracy by incorporating region-specific license plate formats.
- 💾 **Data Export**: Save detection data (plate number, timestamp) to a CSV file or database.
