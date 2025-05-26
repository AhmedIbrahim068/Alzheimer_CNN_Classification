# Alzheimer’s Disease Classification from Brain MRI using CNN

This project uses Convolutional Neural Networks (CNN) to classify MRI brain scans into four stages of Alzheimer’s Disease:
- NonDemented
- VeryMildDemented
- MildDemented
- ModerateDemented

## 👨‍🔬 Team
- Ahmed Ibrahim Khalifa
- Caren Hany Albert

## 📌 Problem Statement
Early and accurate diagnosis of Alzheimer’s Disease is vital for effective treatment. Manual MRI diagnosis is time-consuming and error-prone. This project automates classification using deep learning.

## 🧠 Dataset
The dataset is publicly available from Kaggle and contains MRI images divided into 4 categories. The data was resized, normalized, and stratified into train/val/test sets.

## 🛠️ Tools & Libraries
- Python, TensorFlow, Keras, NumPy, Pandas, Matplotlib, Seaborn, scikit-learn

## 🏗️ CNN Model Architecture
- Conv2D(32) → MaxPooling → Conv2D(64) → MaxPooling → Flatten → Dense(128) → Dropout(0.3) → Output(Dense 4, softmax)

## ⚖️ Handling Class Imbalance
- `class_weight` was applied during training to balance performance across categories.

## 📈 Results
- Validation Accuracy: **~95%**
- Confusion matrix showed strong classification performance, except for some confusion between `Mild` and `Moderate`.

## 🛠️ Challenges & Solutions

| Issue | How we solved it |
|-------|------------------|
| Dataset folders not consistent | Verified structure manually, used only OriginalDataset |
| Some images unreadable | Skipped corrupted files in preprocessing |
| Class imbalance | Used `class_weight` from `sklearn.utils` |
| Dataset structure confusion (augmented vs original) | Created a ZIP with only clean original data |
| `parquet` dataset | Not used due to format mismatch (not image files) |
| Unequal image count per class | Addressed via class weighting, and verified stratified splits |

## 🔍 Evaluation
- Classification report with precision, recall, F1-score
- Visuals: Accuracy/Loss curves and confusion matrix heatmap

## 📂 Project Structure

```
├── model_training.ipynb
├── results/
│   ├── accuracy_loss_plot.png
│   └── confusion_matrix.png
├── README.md
└── requirements.txt
```

## ✅ How to Run
1. Load dataset to `/mnt/data/extracted_dataset_v3/OriginalDataset`
2. Run `model_training.ipynb`
3. Review evaluation metrics and plots in `/results`

## 📄 License
Free for academic use.
