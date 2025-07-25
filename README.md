🩺 Skin Cancer Detection using Deep Learning & Explainable AI
This project focuses on automated skin cancer classification using state-of-the-art deep learning models and explainable AI techniques. Leveraging DenseNet121 with Grad-CAM, the system achieves high accuracy and provides visual explanations to ensure clinical trust and transparency.

Result: Achieved 94.57% training accuracy, 83.03% validation accuracy, and 85.7% testing accuracy on the HAM10000 dataset, with a validation loss of 0.4747.

🔥 Key Features
Multi-class classification for 7 skin cancer types

Deep Learning Models: Basic CNN, ResNet50, MobileNetV2, DenseNet121

Explainable AI: Grad-CAM visualizations for model interpretability

Preprocessing & Augmentation: Resizing, normalization, rotation, flipping, zooming

Detailed Evaluation: Accuracy, precision, recall, F1-score, confusion matrix

🧠 Dataset
Dataset Used: HAM10000 Skin Cancer Dataset

Contains dermatoscopic images of 7 different skin lesion types.

Classes include: akiec, bcc, bkl, df, mel, nv, vasc.

⚙️ Workflow
Data Preprocessing

Resizing, normalization, label encoding

Data Augmentation

Rotation, zoom, flipping, shifting

Model Training

Basic CNN (baseline)

ResNet50 (pretrained, fine-tuned)

MobileNetV2 (lightweight fast model)

DenseNet121 + Grad-CAM (best performance)

Evaluation

Metrics: Accuracy, Precision, Recall, F1-score

Confusion matrix & visual results

Explainability

Grad-CAM heatmaps for model interpretability

📊 Results
Model Performance Comparison
Model	Train Accuracy	Val Accuracy	Train Loss	Val Loss
Basic CNN	81.85%	68.85%	0.4792	0.8857
ResNet50	82.02%	68.85%	0.4699	0.8857
MobileNetV2	70.57%	72.50%	0.7899	0.7386
DenseNet121 + Grad-CAM	94.57%	83.03%	0.2050	0.4747

Classification Report (DenseNet121)
Precision: 0.81

Recall: 0.81

F1-score: 0.79

Weighted Avg Accuracy: 85.7%

Visualizations:
Training & validation accuracy/loss curves

Grad-CAM heatmaps for lesion explainability

Validation accuracy comparison chart across models

(Add images from your poster here, e.g., confusion matrix, Grad-CAM heatmaps, training curves)

🛠 Tech Stack
Languages: Python

Frameworks: TensorFlow, Keras

Libraries: NumPy, Pandas, Matplotlib, OpenCV, Scikit-learn

Tools: Google Colab / Jupyter Notebook

🚀 How to Run
bash
Copy
Edit
git clone https://github.com/yashikasharma2004/SKIN_CANCER_DETECTION--DEEP_LEARNING-.git
cd SKIN_CANCER_DETECTION--DEEP_LEARNING-
pip install -r requirements.txt
python train.py
📌 Future Scope
Integration into web/mobile application for real-time diagnosis

Use of advanced pretrained models (EfficientNet, Vision Transformers)

Cloud deployment for remote healthcare solutions

👩‍💻 Author
Yashika Sharma
B.Tech CSE, Thapar University

📄 License
This project is licensed under the MIT License.
