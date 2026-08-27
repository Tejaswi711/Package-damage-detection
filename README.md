#  Package Damage Detection using Keras & YOLOv8

A deep learning project for detecting whether a package is **damaged** or **intact** using image classification.

This project explores two deep learning approaches:

* **VGG16 Transfer Learning using TensorFlow/Keras**
* **YOLOv8 Image Classification using Ultralytics**

The project also supports predictions on uploaded images, camera-captured images, and videos.

---

## Project Objective

The objective of this project is to automatically classify packages into two categories:

* `damaged`
* `intact`

This can be useful for automated package inspection, logistics, warehouse quality control, and delivery operations.

---

## Approaches Used

### 1. VGG16 Transfer Learning

A pretrained **VGG16** model with ImageNet weights is used as the base model.

The pretrained convolutional layers are frozen, and additional layers are added for binary classification:

```text
VGG16 (ImageNet)
      ↓
Global Average Pooling
      ↓
Dense Layer (128)
      ↓
Dropout (0.5)
      ↓
Sigmoid Output
      ↓
Damaged / Intact
```

The model uses:

* Input size: `224 × 224 × 3`
* GlobalAveragePooling2D
* Dense layer with 128 neurons
* Dropout = 0.5
* Sigmoid output
* Adam optimizer
* Binary Cross-Entropy loss

The VGG16 base model is loaded with ImageNet weights and frozen before training.

---

### 2. CNN From Scratch

A custom CNN was also implemented for comparison.

Architecture:

```text
Input Image
    ↓
Rescaling
    ↓
Conv2D (32)
    ↓
MaxPooling
    ↓
Conv2D (64)
    ↓
MaxPooling
    ↓
Conv2D (128)
    ↓
MaxPooling
    ↓
Flatten
    ↓
Dense (128)
    ↓
Dropout
    ↓
Sigmoid Output
```

The scratch CNN was trained for 15 epochs.

---

### 3. YOLOv8 Classification

The project also uses **YOLOv8** for image classification.

The YOLO model is initialized using:

```text
yolov8n-cls.pt
```

and trained for 15 epochs with an image size of `224 × 224`.

The best trained YOLO model is saved as:

```text
yolov8_cls_best.pt
```

---

##  Dataset

The dataset contains two classes:

```text
dataset/
├── damaged/
└── intact/
```

Before training, images are checked for validity. Invalid or corrupted images are removed.

The dataset is divided into training, validation, and test data for the Keras model.

### Keras Dataset Split

```text
70% → Training
15% → Validation
15% → Testing
```

The notebook creates the training dataset using a 70/30 split and then divides the remaining 30% equally between validation and testing.

---

## Image Preprocessing

Images are resized to:

```text
224 × 224
```

Batch size:

```text
32
```

Random seed:

```text
42
```

The Keras dataset loader uses binary labels because the project contains two classes.

---

##  Class Weighting

Class weights are calculated based on the number of images in the `damaged` and `intact` classes.

This helps reduce the effect of class imbalance during training.

```text
class_weight
```

is passed during model training for both the pretrained model and the scratch CNN.

---

## Model Evaluation

The trained Keras model is evaluated using:

* Accuracy
* Loss
* Precision
* Recall
* F1-score
* Confusion Matrix

The notebook generates a classification report and confusion matrix on the test dataset.

---

## Image Prediction

A trained Keras model can be used to classify a new package image.

The prediction returns:

```text
Class Label
Confidence Score
```

Example:

```text
damaged
92.35%
```

The predicted label and confidence are also displayed on the image.

---

##  Camera Capture

The project includes a Google Colab camera capture function.

The camera can be opened from the notebook, allowing the user to capture a package image.

The captured image is then passed to the YOLOv8 classification model for prediction.

The output displays:

```text
Prediction: damaged
Confidence: 0.93
```

---

##  Video Prediction

The project can also process an uploaded video.

OpenCV is used to:

1. Read the video frame by frame
2. Run YOLO classification
3. Display the predicted class
4. Display confidence
5. Add the prediction to the video
6. Save the processed video

To improve processing speed, YOLO inference is performed every **5th frame**, while the previous prediction is reused for the other frames.

The output video is saved as:

```text
output_detected.mp4
```

---

## Project Workflow

```text
Package Images
      ↓
Data Validation
      ↓
Remove Invalid Images
      ↓
Train / Validation / Test Split
      ↓
Class Weight Calculation
      ↓
       ┌──────────────────────┐
       │                      │
       ↓                      ↓
 VGG16 Transfer          CNN From Scratch
 Learning
       │                      │
       └──────────┬───────────┘
                  ↓
            Model Evaluation
                  ↓
             YOLOv8 Training
                  ↓
        YOLOv8 Classification
                  ↓
      ┌───────────┼───────────┐
      ↓           ↓           ↓
    Image       Camera      Video
 Prediction    Capture    Prediction
```

---

## Technologies Used

* Python
* TensorFlow
* Keras
* VGG16
* YOLOv8
* Ultralytics
* OpenCV
* NumPy
* Pandas
* Matplotlib
* Scikit-learn
* Google Colab
* Google Drive
* Pillow

---

##  Project Structure

```text
Package-Damage-Detection/
│
├── Package_Damage_Detection_Keras_YOLO_Live.ipynb
│
├── models/
│   ├── vgg16_package_damage.keras
│   └── yolov8_cls_best.pt
│
├── dataset/
│   ├── damaged/
│   └── intact/
│
├── output/
│   └── output_detected.mp4
│
├── requirements.txt
│
└── README.md
```

> The exact model and output filenames may be different depending on where the notebook is executed. The notebook currently saves the trained models to a Google Drive directory.

---

##  How to Run

### Option 1 — Google Colab

The notebook is designed to work with Google Colab.

1. Open the notebook in Google Colab.
2. Mount Google Drive.
3. Place the dataset ZIP file in Google Drive.
4. Run the dataset extraction cells.
5. Run image validation.
6. Train the Keras models.
7. Train the YOLOv8 classifier.
8. Run image, camera, or video prediction.

The notebook mounts Google Drive and extracts the dataset from an `archive.zip` file.

---

### Option 2 — Local Environment

Clone the repository:

```bash
git clone https://github.com/your-username/Package-Damage-Detection.git
cd Package-Damage-Detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Then open:

```text
Package_Damage_Detection_Keras_YOLO_Live.ipynb
```

and run the notebook.

---

##  Model Files

The project uses two main trained models.

### Keras Model

The Keras model contains:

```text
VGG16 + Global Average Pooling
+ Dense + Dropout + Sigmoid
```

### YOLO Model

The YOLO model is based on:

```text
YOLOv8n Classification
```

The trained best model is saved as:

```text
yolov8_cls_best.pt
```

---

## Prediction Output

For each image, the model produces:

```text
Class:
damaged / intact

Confidence:
0–100%
```

For videos, the prediction and confidence are displayed directly on the processed frames.

---

##  Key Features

* Binary package damage classification
* Damaged vs intact detection
* VGG16 transfer learning
* Custom CNN implementation
* YOLOv8 classification
* Class weighting
* Camera-based prediction
* Video prediction
* Confidence score
* Confusion matrix
* Classification report
* Model saving and loading

---

##  Evaluation Metrics

The project uses the following metrics:

### Accuracy

Measures the overall percentage of correctly classified images.

### Precision

Measures how many images predicted as a particular class were actually that class.

### Recall

Measures how many actual images of a class were correctly detected.

### F1-Score

The harmonic mean of precision and recall.

### Confusion Matrix

Shows the number of:

* True Positives
* True Negatives
* False Positives
* False Negatives

---

##  Limitations

* The model is trained specifically for the available `damaged` and `intact` package images.
* Performance may vary on images with different lighting, backgrounds, package types, or camera angles.
* The live camera functionality in the notebook is designed for Google Colab.
* YOLO processing every 5th video frame is used to reduce processing time.

---

##  Future Improvements

* Add more package damage categories such as:

  * Broken
  * Torn
  * Crushed
  * Wet
  * Opened
* Increase dataset size and diversity.
* Apply additional data augmentation.
* Fine-tune the VGG16 layers.
* Compare VGG16 with MobileNetV2, ResNet50, and EfficientNet.
* Improve real-time video inference speed.
* Build a standalone web application.
* Deploy the model using Streamlit or another cloud platform.

---

##  Author

**Tejaswi Siddagoni**

If you found this project useful, consider giving the repository a .
