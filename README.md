# Crowd Detection Using YOLOv8

## 📌 Project Overview
This project uses **YOLOv8** for real-time crowd detection in videos. It tracks individuals, groups them based on proximity, and detects crowds that meet a defined threshold.

## 🚀 Features
- Detects individuals using **YOLOv8**.
- Tracks people across frames and assigns unique IDs.
- Identifies crowd formations based on predefined distance thresholds.
- Saves crowd count per frame in a CSV file.

## 📂 Installation
### 1️⃣ Clone the Repository
```sh
git clone https://github.com/yourusername/crowd-detection.git
cd crowd-detection
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Crowd Detection
```sh
python crowd_detection.py
```

## 🔧 Configuration
- **CROWD_THRESHOLD**: Minimum number of people required to be considered a crowd.
- **DISTANCE_THRESHOLD**: Maximum distance between people for them to be considered in the same group.
- **FRAME_THRESHOLD**: Number of frames a person remains tracked before being removed.

## 📊 Output
- **Live Video Stream**: Displays detected individuals and crowds.
- **CSV Output**: A file `crowd_detection_results.csv` is generated containing:

| Frame Number | Crowd Count |
|-------------|-------------|
| 100         | 5           |
| 250         | 7           |

## 📌 Dependencies
- **Python 3.8+**
- **OpenCV**
- **YOLOv8 (Ultralytics)**
- **NumPy**
- **Pandas**
- **SciPy**

## 📜 License
This project is licensed under the **MIT License**.

## ✨ Acknowledgments
- **Ultralytics YOLOv8** for object detection.
- **OpenCV** for image processing.
- **SciPy** for spatial distance calculations.

---

### **requirements.txt**
```txt
opencv-python
opencv-python-headless
torch
torchvision
numpy
pandas
scipy
ultralytics
```

