
 Brain Tumor Detection using YOLO11 and SAM2_Model

 📌 Overview
This project focuses on detecting brain tumors using **YOLO** object detection and **Segment Anything Model (SAM)** for precise segmentation. The model is trained on a dataset sourced from **Roboflow**, and inference is performed on test images for detailed analysis.

 🚀 Setup
Before running the code, ensure you have installed the required dependencies:

```sh
!pip install ultralytics roboflow
```

 🧠 Dataset
The dataset is fetched from **Roboflow** using the API:

```python
from roboflow import Roboflow

# Access dataset using API
rf = Roboflow(api_key="JZYnqR9HmLD000Lip7vh")
project = rf.workspace("brain-tumor-detection-wsera").project("tumor-detection-ko5jp")
version = project.version(8)
```

 🏗️ Model Training (YOLO11n)
The YOLO model is trained using the dataset obtained from **Roboflow**:

```python
from ultralytics import YOLO

# Load YOLO11n model
model = YOLO("yolo11n.pt")

# Train the model
train_results = model.train(
    data="https://universe.roboflow.com/ds/K764Jv4Jnk?key=lad0tDWoNc", 
    epochs=20,  # Number of training epochs
    imgsz=640,  # Image size for training
    device=0
)
```

 🎯 Inference (Object Detection)
Once trained, the YOLO model performs object detection on test images:

```python
from ultralytics import YOLO

# Load trained model
model = YOLO("/kaggle/working/runs/detect/train/weights/best.pt")

# Run inference on a single image
results = model("/kaggle/working/datasets/K764Jv4Jnk/test/images/meningioma_1022_jpg.rf.a3ae957a204e1f240de0d48f7c95c0aa.jpg", save=True)

# Process results and extract bounding boxes
for result in results:
    boxes = result.boxes  # Bounding box outputs
    print(boxes)
```

For batch processing of all images in the directory:

```python
results = model.predict(source="/kaggle/working/datasets/K764Jv4Jnk/test/images", save=True)
```

🔍 Segmentation (SAM Model)
After detection, SAM is used for refined segmentation:

```python
from ultralytics import SAM

# Load the SAM model
sam_model = SAM("sam2_b.pt")

# Segment YOLO detections using SAM
for result in results:
    class_ids = result.boxes.cls.int().tolist()  # Object class IDs
    if len(class_ids):
        boxes = result.boxes.xyxy   # Bounding box outputs
        sam_results = sam_model(result.orig_img, bboxes=boxes, verbose=False, save=True, device=0)
```

📊 Results
- **YOLO** detects potential tumors and provides bounding boxes.
- **SAM** refines the segmentation, improving accuracy.
- The detected images with bounding boxes and segmentation masks are saved for further analysis.

🚀 Future Improvements
- Fine-tune the YOLO model for improved detection accuracy.
- Experiment with different SAM configurations for enhanced segmentation.
- Implement automated reporting for medical professionals.

 🤝 Contributions
Feel free to **fork** this repository, suggest improvements, and submit **pull requests**!

🔖 License
This project is licensed under the **MIT License**.
