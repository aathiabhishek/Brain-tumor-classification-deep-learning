# 🧠 Brain Tumor Detection & Classification Using Deep Learning

An end-to-end deep learning computer vision application for **brain MRI image classification**. The project develops, tunes, evaluates, and compares six deep learning architectures for classifying MRI images into four categories: **Glioma, Meningioma, Pituitary Tumor, and No Tumor**.

The best-performing model is integrated into an interactive **Streamlit application** that allows users to upload MRI images, select different trained models, view predictions and confidence scores, explore educational tumor information, and access general nutrition resources.

> ⚠️ **Medical Disclaimer:** This project is intended for educational, research, and demonstration purposes only. Model predictions are not medical diagnoses and should not replace evaluation by a qualified healthcare professional or radiologist.

---

## 📌 Project Highlights

* 🧠 **Multi-model deep learning benchmark** across six CNN-based architectures.
* 🔬 **Four-class brain MRI classification**:

  * Glioma Tumor
  * Meningioma Tumor
  * Pituitary Tumor
  * No Tumor
* ⚙️ **Hyperparameter tuning** using Keras Tuner.
* 🔄 **Transfer learning** using VGG16, MobileNetV2, and ResNet50.
* 📊 Evaluation using accuracy, precision, recall, and F1-score.
* 🌐 Interactive **Streamlit web application**.
* 🔐 Secure login and session-based authentication.
* 📤 MRI image upload and real-time preprocessing.
* 🤖 User-selectable trained models.
* 📈 Prediction confidence and class probability visualization.
* 📄 Downloadable prediction reports.
* 🧬 Educational tumor information from trusted medical sources.
* 🍎 General nutrition and food information.
* 📱 Responsive and user-friendly dashboard interface.

---

# 🏗️ System Workflow

```text
                    ┌─────────────────────┐
                    │      User Login     │
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

## 📊 Dataset

This project uses the **Brain Tumor MRI Dataset** sourced from Kaggle. The dataset contains brain MRI images across four classification categories:

* 🧬 **Glioma**
* 🧬 **Meningioma**
* 🧠 **Pituitary**
* ✅ **No Tumor**

### Dataset Distribution

| Classification |    Images |
| -------------- | --------: |
| 🧬 Glioma      |     1,044 |
| 🧬 Meningioma  |     1,039 |
| 🧠 Pituitary   |     1,045 |
| ✅ No Tumor     |       500 |
| **Total**      | **3,628** |

### Dataset Split

| Dataset   |    Images |
| --------- | --------: |
| Training  |     2,870 |
| Testing   |       758 |
| **Total** | **3,628** |

### Dataset Source

The original dataset is available through Kaggle:

**Brain Tumor MRI Dataset**
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

The dataset is used exclusively for **educational and research-oriented machine learning experimentation**.

### Dataset Availability

The original MRI dataset is **not included in this repository** due to its size and dataset distribution considerations.

To reproduce the experiments:

1. Download the dataset directly from the original Kaggle source.
2. Extract the dataset locally.
3. Place the required image folders inside the project's dataset directory.
4. Follow the preprocessing and training workflow provided in the Jupyter notebook.

Users are responsible for reviewing and complying with the **Kaggle dataset's applicable license and usage terms**.

### Data Privacy

No patient-identifying information is intentionally included in this repository. The project is designed for educational and research purposes and should not be used as a substitute for clinical diagnosis.


---

# ⚙️ Image Preprocessing

The inference pipeline is designed to match the preprocessing used during model training.

### Preprocessing steps

1. Load the uploaded MRI image.
2. Convert the image to RGB.
3. Resize the image to:

```text
224 × 224 × 3
```

4. Convert the image into a NumPy array.
5. Normalize pixel values:

```python
pixel_value / 255.0
```

6. Add the batch dimension:

```text
(224, 224, 3)
        ↓
(1, 224, 224, 3)
```

The application intentionally avoids additional preprocessing techniques such as CLAHE, denoising, or center cropping during inference because these were not part of the training preprocessing pipeline.

---

# 🧠 Deep Learning Models

Six different architectures were developed and evaluated.

## 1. 🧱 Custom CNN

A convolutional neural network developed from scratch and used as the baseline architecture.

The network uses progressive convolutional feature extraction with filter sizes:

```text
32 → 64 → 128 → 256 → 512
```

The architecture incorporates components such as:

* `Conv2D`
* `BatchNormalization`
* `MaxPooling2D`
* Dense layers
* Softmax output layer

---

## 2. ⚙️ Hyperparameter-Tuned CNN

A CNN optimized using **Keras Tuner**.

Hyperparameter experimentation was used to identify a suitable architecture and training configuration.

The tuned model uses:

* Multiple convolutional blocks
* 96-filter convolution layers
* Spatial dropout
* Adam optimizer
* Learning rate of `0.001`
* Four-class Softmax classification

---

## 3. 📦 Autoencoder Classifier

An autoencoder-based architecture designed to learn compact latent representations of MRI images.

The learned representation is then used for classification.

The architecture focuses on:

```text
MRI Image
     ↓
Encoder
     ↓
Latent Representation
     ↓
Classification Layers
     ↓
4-Class Prediction
```

This approach provides an alternative representation-learning strategy compared with conventional CNN classifiers.

---

## 4. 🔷 VGG16

A transfer-learning architecture based on **VGG16** pretrained on ImageNet.

The pretrained convolutional layers are used as feature extractors, followed by task-specific classification layers.

Advantages include:

* Strong feature extraction capability
* Established architecture
* Effective transfer learning baseline

A major limitation is its comparatively large parameter count and computational requirements.

---

## 5. ⚡ MobileNetV2

A lightweight CNN architecture designed for efficient image processing.

MobileNetV2 uses depthwise separable convolutions and inverted residual structures to reduce computational requirements.

The model provides a strong balance between:

```text
Accuracy
    +
Inference Efficiency
    +
Model Size
```

Based on the project's evaluation, **MobileNetV2 achieved the strongest overall performance among the tested architectures**.

---

## 6. 🔗 ResNet50

A deep residual neural network based on residual/skip connections.

The architecture enables deeper feature learning while helping address optimization difficulties associated with very deep networks.

ResNet50 was included as an additional transfer-learning benchmark.

---

# 📈 Model Performance

The trained models were evaluated on an independent test set using multiple classification metrics.

| Model              | Accuracy | Precision |   Recall | F1-Score | Parameters |
| ------------------ | -------: | --------: | -------: | -------: | ---------: |
| 🏆 **MobileNetV2** | **0.94** |  **0.93** | **0.94** | **0.93** |   **3.4M** |
| 🔍 **Tuned CNN**   |     0.88 |      0.87 |     0.88 |     0.87 |       2.1M |
| 🔷 **VGG16**       |     0.85 |      0.84 |     0.85 |     0.84 |       138M |
| 📦 **Autoencoder** |     0.85 |      0.83 |     0.85 |     0.84 |       1.8M |
| 🧱 **CNN Scratch** |     0.83 |      0.82 |     0.83 |     0.82 |       5.0M |
| ⚠️ **ResNet50**    |     0.60 |      0.62 |     0.60 |     0.59 |      25.6M |

### 🏆 Best Performing Model

**MobileNetV2** achieved the highest overall test performance in this project:

```text
Accuracy   : 94%
Precision  : 93%
Recall     : 94%
F1-Score   : 93%
Parameters : ~3.4M
```

Its combination of classification performance and relatively low parameter count makes it an attractive model for efficient inference.

> **Note:** These metrics represent performance on this project's dataset and test split. They should not be interpreted as clinical performance or evidence of medical diagnostic accuracy.

---

# 🔬 Evaluation Metrics

The models were evaluated using:

### Accuracy

Measures the overall percentage of correctly classified images.

```text
Accuracy =
Correct Predictions / Total Predictions
```

### Precision

Measures how many images predicted as a particular class were actually from that class.

```text
Precision =
True Positives / (True Positives + False Positives)
```

### Recall

Measures how many actual samples from a class were correctly identified.

```text
Recall =
True Positives / (True Positives + False Negatives)
```

### F1-Score

Provides a balance between precision and recall.

```text
F1 =
2 × (Precision × Recall)
    ----------------------
      Precision + Recall
```

For medical image classification, precision and recall are particularly important because an accuracy score alone does not fully describe classification performance.

---

# 🌐 Streamlit Application

The trained models are integrated into an interactive Streamlit application called:

## 🧠 NeuroScan Diagnostics

The application provides an end-to-end interface for experimenting with the trained MRI classification models.

### 🔐 Secure Authentication

Users must authenticate before accessing the dashboard.

```text
Username
   +
Password
   ↓
Authentication
   ↓
Dashboard
```

Authentication is handled through:

```text
utils/auth.py
```

---

# 📤 MRI Upload & Analysis

Users can upload:

```text
.jpg
.jpeg
.png
```

The application then:

1. Reads the image.
2. Converts it to RGB.
3. Saves the original image.
4. Resizes it to `224 × 224`.
5. Normalizes the image.
6. Displays the processed model input.
7. Passes the image to the selected model.
8. Generates the four-class prediction.
9. Calculates prediction confidence.
10. Displays class probabilities.

---

# 🤖 Model Selection

Users can select between the six trained architectures:

```text
CNN
CNN (Hypertuned)
VGG16
Autoencoder
MobileNetV2
ResNet50
```

Models are loaded dynamically from the `models/` directory.

TensorFlow/Keras model caching is used to reduce unnecessary model reloads during Streamlit reruns.

---

# 📊 Prediction Results

After processing an MRI scan, the dashboard displays:

* Predicted classification
* Confidence score
* Probability for each class
* Selected model
* Processed image
* Model input shape
* Low-confidence warning
* Downloadable prediction report

Example:

```text
Prediction:
Glioma Tumor

Confidence:
94.32%

Class Probabilities:

Glioma Tumor       94.32%
Meningioma Tumor    2.13%
No Tumor            1.02%
Pituitary Tumor     2.53%
```

If confidence is below the configured threshold, the application displays a warning to indicate that the model prediction may be less reliable.

---

# 🧬 Tumor Information

The **Tumor Info** section provides educational information for:

* Glioma
* Meningioma
* Pituitary Tumor
* No Tumor

Information may include:

* General description
* Common symptoms
* Possible effects
* Diagnostic approaches
* Treatment overview
* Educational resources
* Trusted medical references

The application retrieves information from trusted medical resources such as the **National Cancer Institute (NCI)**.

---

# 🍎 Nutrition Resources

The **Diet Plan** section provides general food and nutrition information associated with the selected classification.

Food information is stored locally in:

```text
data/food_data.json
```

The application can display information such as:

* Food name
* Nutrients
* Calories
* General benefits

> ⚠️ Nutrition information is general educational content and is not intended to provide individualized medical or dietary treatment recommendations.

---

# 📄 Prediction Report

After an analysis, users can download a text-based report containing:

```text
NeuroScan Diagnostics Report

Prediction
Confidence
Model Used
Input Image
Image Size

Medical/Educational Disclaimer
```

This provides a convenient record of the model inference result for demonstration and project evaluation purposes.

---

# 🗂️ Project Structure

```text
Brain-Tumor-Detection/
│
├── app.py
│
├── models/
│   ├── brain_tumor_model_cnn.h5
│   ├── brain_tumor_model_tuned.h5
│   ├── brain_tumor_model_vgg16.h5
│   ├── brain_tumor_model_autoencoder.h5
│   ├── brain_tumor_model_mobilenetv2.h5
│   └── brain_tumor_model_resnet50.h5
│
├── data/
│   └── food_data.json
│
├── uploads/
│
├── processed_images/
│
├── utils/
│   └── auth.py
│
├── requirements.txt
│
└── README.md
```

---

# 🛠️ Technology Stack

### Programming Language

* Python

### Deep Learning

* TensorFlow
* Keras
* Keras Tuner

### Machine Learning

* NumPy
* Scikit-learn

### Image Processing

* Pillow

### Web Application

* Streamlit

### Web Scraping / Information Retrieval

* Requests
* BeautifulSoup

### Authentication

* Custom Python authentication module

### Data / Configuration

* JSON

---

# 🚀 Getting Started

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Brain-Tumor-Detection
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the actual repository URL.

---

## 2. Create a Virtual Environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

If TensorFlow is not already installed:

```bash
pip install tensorflow
```

---

## 4. Verify Model Files

Make sure the trained models are available inside:

```text
models/
```

Expected files:

```text
brain_tumor_model_cnn.h5
brain_tumor_model_tuned.h5
brain_tumor_model_vgg16.h5
brain_tumor_model_autoencoder.h5
brain_tumor_model_mobilenetv2.h5
brain_tumor_model_resnet50.h5
```

---

## 5. Verify Food Data

Make sure the following file exists:

```text
data/food_data.json
```

---

## 6. Run the Streamlit Application

```bash
streamlit run app.py
```

The application will open in your browser.

---

# 🔑 Application Workflow

```text
1. Login
   ↓
2. Open Upload MRI
   ↓
3. Select Deep Learning Model
   ↓
4. Upload MRI Image
   ↓
5. Preprocess Image
   ↓
6. Run Model Inference
   ↓
7. View Classification
   ↓
8. Review Confidence & Probabilities
   ↓
9. Download Report
   ↓
10. Explore Tumor Information
   ↓
11. Explore General Nutrition Resources
```

---

# 🔐 Security & Data Handling

The application includes a basic authentication layer to restrict access to the dashboard.

Uploaded images are stored in the application's:

```text
uploads/
```

Processed model inputs are stored in:

```text
processed_images/
```

For a real-world medical deployment, additional security controls would be required, including:

* Encryption
* Secure credential management
* Access control
* Audit logging
* Secure storage
* Data retention policies
* Privacy compliance
* Protected health information handling

This project is **not intended for production clinical deployment**.

---

# ⚠️ Medical Disclaimer

NeuroScan Diagnostics is an **educational and research-oriented deep learning project**.

The predictions generated by the models:

* Are not medical diagnoses.
* Have not been validated for clinical use.
* Should not be used to make treatment decisions.
* Should not replace radiological interpretation.
* Should not replace consultation with a qualified healthcare professional.

MRI images and model predictions should always be evaluated by appropriately qualified medical professionals.

---

# 📌 Limitations

Several limitations should be considered when interpreting the project results:

* The dataset is relatively small for deep learning.
* Dataset performance may not generalize to external MRI datasets.
* MRI acquisition protocols can vary between hospitals and imaging systems.
* Dataset bias may affect model performance.
* Test-set performance does not guarantee real-world clinical performance.
* No external clinical validation was performed.
* The models were developed for educational/research purposes.
* The application does not perform formal radiological diagnosis.

---

# 🔮 Future Improvements

Potential future improvements include:

* 🔬 External dataset validation
* 🧠 Explainable AI using Grad-CAM
* 📊 Confusion matrix visualization inside the dashboard
* 📈 ROC-AUC and precision-recall curves
* 🖼️ MRI segmentation before classification
* 🧬 Tumor localization and bounding boxes
* 🏥 Multi-institutional validation
* ⚡ Model quantization and optimization
* 📱 Mobile-friendly inference
* 🔐 Improved authentication and authorization
* ☁️ Cloud deployment
* 🗄️ Database-backed prediction history
* 📑 PDF report generation
* 🧠 Ensemble model prediction
* 🔍 Model comparison directly inside the dashboard

---

# 🎯 Key Learning Outcomes

This project demonstrates practical experience with:

* Deep learning for computer vision
* CNN architecture design
* Transfer learning
* Hyperparameter tuning
* Multi-class image classification
* Image preprocessing
* Model evaluation
* TensorFlow/Keras
* Model serialization and loading
* Streamlit application development
* Authentication
* File handling
* Real-time inference
* Web-based ML deployment
* Model comparison and benchmarking
* Building an end-to-end AI application

---

# 👨‍💻 Project Summary

**Brain Tumor Detection & Classification Using Deep Learning** combines computer vision, deep learning, model optimization, and web application development into a complete end-to-end AI project.

The project evaluates six different architectures and demonstrates how model architecture affects the balance between classification performance and computational efficiency.

Among the evaluated architectures, **MobileNetV2 achieved the strongest overall performance**, making it the selected model for efficient inference experimentation within the NeuroScan Diagnostics application.

The project demonstrates the complete workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
Model Development
   ↓
Hyperparameter Tuning
   ↓
Transfer Learning
   ↓
Model Evaluation
   ↓
Model Comparison
   ↓
Streamlit Deployment
   ↓
Interactive AI Application
```

---

## ⭐ Author

**Aathi Abhishek Thiagarajan**

Machine Learning / AI Developer

---

## ⚕️ Final Note

This repository is intended to demonstrate **deep learning, computer vision, and AI application development skills**. It should not be interpreted as a clinically validated brain tumor diagnostic system.
