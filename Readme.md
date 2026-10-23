# 🧠 Brain Tumor Detection & Classification

An end-to-end **Deep Learning computer vision project** that classifies brain MRI images into four categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**.

The project compares six deep learning architectures and deploys them through an interactive **Streamlit application** for real-time MRI image classification.

> ⚠️ **Medical Disclaimer:** This project is for educational and research purposes only. Predictions are not medical diagnoses and should not be used for clinical decision-making.

---

## 🌐 Live Demo

🚀 **[Try NeuroScan Diagnostics](https://brain-tumor-classification-models.streamlit.app/)**

Upload an MRI image, select a trained model, view predictions and confidence scores, explore educational information, and download prediction reports.

---

## 🎯 Project Highlights

* 🧠 Four-class brain MRI classification
* 🔬 Six deep learning architectures compared
* ⚙️ CNN hyperparameter tuning with Keras Tuner
* 🔄 Transfer learning with VGG16, MobileNetV2 & ResNet50
* 📊 Accuracy, Precision, Recall & F1 evaluation
* 🌐 Interactive Streamlit dashboard
* 📤 Real-time MRI image inference
* 📈 Class probability and confidence visualization
* 📄 Downloadable prediction reports
* 🔐 Authentication layer
* ☁️ Streamlit Community Cloud deployment

---

# 🏗️ System Workflow

```text
                    ┌─────────────────────┐
                    │      Home    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Upload MRI Image │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Image Preprocessing │
                    │    224 × 224 × 3    │
                    │     Rescale /255    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   Select AI Model   │
                    └──────────┬──────────┘
                               │
             ┌─────────────────┼─────────────────┐
             │                 │                 │
             ▼                 ▼                 ▼
        Custom CNN        Tuned CNN        Autoencoder
             │                 │                 │
             ├─────────────────┼─────────────────┤
             │                 │                 │
             ▼                 ▼                 ▼
           VGG16          MobileNetV2         ResNet50
             │                 │                 │
             └─────────────────┼─────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ 4-Class Prediction  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Prediction &        │
                    │ Confidence Score    │
                    └──────────┬──────────┘
                               │
                 ┌─────────────┼─────────────┐
                 ▼             ▼             ▼
            Tumor Info     Diet Plan     Report Download
```

---

## 🧠 Models

| Model       | Approach                  |
| ----------- | ------------------------- |
| Custom CNN  | CNN from scratch          |
| Tuned CNN   | Keras Tuner optimized CNN |
| Autoencoder | Representation learning   |
| VGG16       | Transfer Learning         |
| MobileNetV2 | Transfer Learning         |
| ResNet50    | Transfer Learning         |

---

## 📊 Model Performance

| Model              | Accuracy | Precision |  Recall |      F1 |
| ------------------ | -------: | --------: | ------: | ------: |
| 🏆 **MobileNetV2** |  **94%** |   **93%** | **94%** | **93%** |
| Tuned CNN          |      88% |       87% |     88% |     87% |
| VGG16              |      85% |       84% |     85% |     84% |
| Autoencoder        |      85% |       83% |     85% |     84% |
| Custom CNN         |      83% |       82% |     83% |     82% |
| ResNet50           |      60% |       62% |     60% |     59% |

**MobileNetV2** achieved the best overall performance with approximately **3.4M parameters**, providing a strong balance between classification performance and model efficiency.

---

## 📂 Dataset

**Brain Tumor MRI Dataset — Kaggle**

| Class      |    Images |
| ---------- | --------: |
| Glioma     |     1,044 |
| Meningioma |     1,039 |
| Pituitary  |     1,045 |
| No Tumor   |       500 |
| **Total**  | **3,628** |

**Train:** 2,870 images
**Test:** 758 images

🔗 [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

The original dataset is not included in this repository.

---

## ⚙️ Preprocessing

MRI images are processed using the same pipeline used during model development:

```text
MRI Image
    ↓
RGB Conversion
    ↓
Resize → 224 × 224
    ↓
Normalize → /255
    ↓
Batch Dimension
    ↓
Model Inference
```

Final input shape:

```text
(1, 224, 224, 3)
```

---

## 🌐 Application Features

**NeuroScan Diagnostics** provides:

* MRI image upload
* AI model selection
* Real-time classification
* Prediction confidence
* Class probability visualization
* Tumor educational information
* General nutrition information
* Downloadable prediction reports
* Authentication

---

## 🗂️ Project Structure

```text
Brain-tumor-classification-deep-learning/
│
├── app.py
├── model.py
├── requirements.txt
├── Readme.md
│
├── models/
│   ├── brain_tumor_model_cnn.h5
│   ├── brain_tumor_model_tuned.h5
│   ├── brain_tumor_model_vgg16.h5
│   ├── brain_tumor_model_autoencoder.h5
│   ├── brain_tumor_model_mobilenetv2.h5
│   ├── brain_tumor_model_resnet50.h5
│   └── class_indices.json
│
├── data/
│   └── food_data.json
│
├── notebook/
│   └── data.ipynb
│
├── results/
│   ├── accuracy_plots/
│   ├── confusion_matrices/
│   ├── loss_plots/
│   └── training_history/

```

---

## 🛠️ Tech Stack

**Python · TensorFlow · Keras · Keras Tuner · NumPy · Scikit-learn · Pillow · Streamlit · Requests · BeautifulSoup · Git · Git LFS**

---

## 🚀 Run Locally

```bash
git clone https://github.com/aathiabhishek/Brain-tumor-classification-deep-learning.git

cd Brain-tumor-classification-deep-learning

pip install -r requirements.txt

streamlit run app.py
```

---

## ☁️ Deployment

The application is deployed using **Streamlit Community Cloud** with GitHub and Git LFS for the trained model files.

🌐 **[Live Application](https://brain-tumor-classification-models.streamlit.app/)**

💻 **[GitHub Repository](https://github.com/aathiabhishek/Brain-tumor-classification-deep-learning)**

---

## ⚠️ Limitations

* Dataset size and diversity are limited.
* No external clinical validation was performed.
* Performance may not generalize to other MRI datasets.
* Predictions are not clinically validated.
* The application is not intended for medical diagnosis.

---

## 🔮 Future Improvements

* Grad-CAM explainability
* MRI tumor segmentation
* External dataset validation
* Model ensemble
* Prediction history
* PDF reports
* Model optimization and quantization

---

## 👨‍💻 Author

**Aathi Abhishek Thiagarajan**

Machine Learning / AI Developer

🔗 **[GitHub](https://github.com/aathiabhishek)**

---

## ⚕️ Medical Disclaimer

This project is intended strictly for **educational, research, and demonstration purposes**. Model predictions are experimental outputs and should not replace professional medical evaluation, radiological interpretation, or clinical diagnosis.
