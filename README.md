# Advanced CNN for Image Classification  

## 📌 Overview  
This repository contains an advanced **Convolutional Neural Network (CNN)** architecture that integrates **pretrained models** with **custom layers** for high-performance image classification.  
The model achieves **state-of-the-art accuracy (99.65% on test set)**, demonstrating the power of hybrid CNNs for real-world computer vision applications.  

---

## 🚀 Features  
- ✅ Hybrid CNN with pretrained + custom layers  
- ✅ Achieved **99.69% validation accuracy** and **99.65% test accuracy**  
- ✅ Multi-scale inference for robust predictions  
- ✅ Configurable training pipeline (AdamW optimizer, early stopping, augmentation)  
- ✅ Research-paper style Jupyter Notebook for reproducibility  

---

## 📊 Results  
- **Validation Accuracy**: 99.69%  
- **Validation Accuracy (final epoch)**: 99.65%  
- **Test Set Accuracy**: 99.65%  

> ⚠️ Note: During inference, ensure that input images exist at the provided path.  
Example error encountered:  
```bash
cv2.error: OpenCV(4.12.0) ... error: (-215:Assertion failed) !_src.empty() in function 'cv::cvtColor'
```
This occurs when the test image file path is invalid.  

---

## 🛠️ Installation  
```bash
git clone https://github.com/yourusername/advanced-cnn.git
cd advanced-cnn
pip install -r requirements.txt
```

---

## 📂 Project Structure  
```
advanced-cnn/
│── advance_cnn_research.ipynb   # Research-style notebook
│── advanced_cnn.py              # Python training & inference script
│── requirements.txt             # Dependencies
│── README.md                    # Project documentation
│── deepfake_dataset/            # Example dataset (real & fake images)
```

---

## ⚡ Usage  

### Train the model  
```bash
python advanced_cnn.py --train
```

### Run inference  
```bash
python advanced_cnn.py --infer --image_path ./deepfake_dataset/real/sample_real.jpg
```

---

## 📖 Methodology  
1. **Data Preprocessing**: Normalization, augmentation, and batching  
2. **Architecture**: Hybrid CNN (pretrained + custom layers)  
3. **Training Strategy**: AdamW optimizer, weight decay, early stopping  
4. **Evaluation**: Accuracy, Precision, Recall, F1-score, confusion matrix  

---

## 🧪 Future Work  
- Integrate **attention mechanisms**  
- Apply **explainability methods** like Grad-CAM  
- Deploy the model via **FastAPI/Streamlit** for real-world use  

---

## 📜 Citation  
If you use this repository, please cite:  
```
@article{krishnaadvancedcnn,
  title={Advanced Convolutional Neural Network for Image Classification},
  author={krishna mishra},
  journal={GitHub Repository},
  year={2025}
}
```
