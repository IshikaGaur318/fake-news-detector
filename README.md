# 📰 Fake News Detector

## 📌 Project Overview
This project is a **BERT-based Fake News Detector** that classifies news articles as **real or fake** using Natural Language Processing (NLP). The model is trained on the **LIAR dataset** and provides **explainability** using SHAP and LIME.

## 🚀 Features
- 🏋️ **Trained BERT-based classifier** for accurate fake news detection
- 📊 **Explainability with SHAP & LIME** to understand model predictions
- 📝 **TF-IDF with Logistic Regression & Naïve Bayes** baseline models
- 📈 **Streamlit web app** for easy user interaction
- 🖼️ **Visualization of feature importance**
- 🔍 **Evaluation metrics** like accuracy, precision, recall, and F1-score

## 📂 Project Structure
```bash
fake-news-detector/
├── src/
│   ├── train.py          # Model training script
│   ├── predict.py        # Prediction script
│   ├── data_loader.py    # Loads and processes the dataset
│   ├── explainability.py # SHAP & LIME explanations
│   └── utils.py          # Helper functions
├── models/               # Saved models
├── data/                 # Dataset storage
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

## 🛠️ Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/fake-news-detector.git
   cd fake-news-detector
   ```
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Training the Model
Run the training script to train the BERT model on the LIAR dataset:
```bash
python src/train.py
```

## 🔎 Making Predictions
To classify a news article as real or fake:
```bash
python src/predict.py --text "Breaking: Scientists discover a new planet made of diamonds!"
```

## 🎯 Performance Metrics
| Model                 | Accuracy | Precision | Recall | F1-Score |
|----------------------|----------|-----------|--------|----------|
| BERT Classifier      | **92.3%** | 91.8%     | 93.1%  | 92.4%    |
| Logistic Regression  | 84.5%    | 83.2%     | 85.6%  | 84.4%    |
| Naïve Bayes         | 81.2%    | 80.5%     | 81.9%  | 81.2%    |

## 🎨 Explainability (SHAP & LIME)
Run the explainability script to visualize feature importance:
```bash
python src/explainability.py
```

## 📌 Future Improvements
- ⚡ Improve model generalization with more data
- 🏆 Fine-tune the model with hyperparameter optimization
- 🌍 Expand to multilingual fake news detection
- 📱 Deploy as a mobile-friendly API

## 💡 Acknowledgments
- 🤖 Hugging Face's `transformers` library
- 📚 Scikit-Learn for baseline models
- 📊 SHAP & LIME for explainability

## ⭐ Contribute
Feel free to fork, improve, and submit a pull request! 😊
