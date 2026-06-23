# Pratik_HindiOCR

# 📝 Hindi Handwritten Character Recognition using ResNet18

A Deep Learning based Hindi (Devanagari) Handwritten Character Recognition system built using **PyTorch**, **ResNet18**, and **Streamlit**.

The project can recognize:

* 36 Hindi Characters
* 10 Hindi Digits
* Total: **46 Classes**

The application allows users to upload an image of a handwritten Hindi character, automatically preprocesses the image, and predicts the corresponding Hindi character.

---

## Features

✅ Upload handwritten Hindi character image

✅ Automatic preprocessing

* Convert RGB → Grayscale
* Thresholding using Otsu Method
* Remove extra background
* Crop character region automatically
* Resize to 64×64
* Normalize image

✅ Deep Learning prediction using ResNet18

✅ Display predicted Hindi character

✅ Display prediction confidence

✅ Web Interface using Streamlit

---

## Dataset

This project uses the **Devanagari Handwritten Character Dataset**.

Classes:

* 36 Hindi Characters
* 10 Hindi Digits

Image Format:

* Grayscale
* 32×32 images
* Around 92,000 handwritten images

Example Classes:

```text
क ख ग घ ङ
च छ ज झ ञ
ट ठ ड ढ ण
त थ द ध न
प फ ब भ म
य र ल व
श ष स ह
क्ष त्र ज्ञ

० १ २ ३ ४ ५ ६ ७ ८ ९
```

---

## Model Architecture

The model is based on:

**ResNet18**

Modifications:

* Input: Grayscale image converted to 3 channels
* Image Size: 64×64
* Output Layer: 46 Classes

Final Layer:

```python
model.fc = nn.Linear(model.fc.in_features, 46)
```

---

## Pretrained Model

The trained model (`devanagari_resnet18.pth`) is **not included in this repository** because of GitHub file size limitations.

### Download Model

Download the pretrained model from Google Drive:

**Google Drive Link:**
https://drive.usercontent.google.com/download?id=1li3vjKVcsmd_LKIqdr2H0zaDIRQfJ7t4&export=download&authuser=0

After downloading, place:

```text
devanagari_resnet18.pth
```

inside:

```text
HindiOCR/

├── app.py
├── preprocess.py

├── devanagari_resnet18.pth
├── class_names.txt

├── requirements.txt
└── README.md
```

---

## Preprocessing Pipeline

When the user uploads an image:

```text
Uploaded Image

↓

Convert to Grayscale

↓

Threshold (Otsu)

↓

Crop Character Region

↓

Resize to 64×64

↓

Normalize

↓

ResNet18 Prediction

↓

Display Hindi Character
```

---

## Project Structure

```text
HindiOCR/

├── app.py
├── preprocess.py

├── class_names.txt

├── requirements.txt
└── README.md
```

---

## Installation

### Clone Repository

```bash
git clone YOUR_REPOSITORY_URL

cd HindiOCR
```

---

### Create Virtual Environment

**Windows**

```bash
python -m venv venv

venv\Scripts\activate
```

**Linux / Mac**

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

or manually:

```bash
pip install torch
pip install torchvision
pip install streamlit
pip install pillow
pip install numpy
pip install opencv-python
```

---

## Requirements

Create:

```text
requirements.txt
```

Add:

```text
torch
torchvision
streamlit
pillow
numpy
opencv-python
```

---

## Run the Application

```bash
streamlit run app.py
```

Browser will open automatically:

```text
http://localhost:8501
```

---

## How To Use

1. Open the Streamlit application.

2. Upload an image:

```text
png
jpg
jpeg
```

3. The application automatically:

* Converts image to grayscale
* Applies thresholding
* Crops character region
* Resizes image to 64×64
* Predicts the Hindi character

4. Output:

```text
Prediction : क

Confidence : 99.18%
```

---

## Training Details

**Model**

```text
ResNet18
```

**Optimizer**

```text
Adam
```

**Learning Rate**

```text
0.001
```

**Loss Function**

```text
CrossEntropyLoss
```

**Epochs**

```text
20
```

**Image Size**

```text
64×64
```

---

## Future Improvements

* Manual Crop Tool
* Drawing Canvas for Hindi Characters
* Word Recognition
* Hindi OCR for complete sentences
* Mobile App Integration
* Deploy on Streamlit Cloud

---

## Author

**Pratik Kumar Sinha**

B.Tech CSE

Hindi Handwritten Character Recognition using Deep Learning and ResNet18.
