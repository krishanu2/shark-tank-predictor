# 🦈 Shark Tank Deal Predictor

This is a Machine Learning + NLP project built using Python, XGBoost, and Gradio that predicts whether a pitch on **Shark Tank US** will receive a deal — based on industry, funding asked, equity offered, valuation, and pitch description.

## 🚀 Live Demo (Optional)
> [Add Gradio link here if deployed]

---

## 📊 Features

- 📁 Trained on actual Shark Tank US data (Season-wise)
- 💬 Natural Language Processing (NLP) on pitch descriptions
- 🤖 Machine Learning powered by XGBoost Classifier
- 🧠 Text Preprocessing using NLTK (stopwords, lowercase, etc.)
- 🎛️ Simple and clean Gradio UI
- 📈 Model Evaluation with Precision, Recall, F1-score

---

## 🧠 Model Details

The model uses:
- `industry`, `original_ask_amount`, `original_offered_equity`, `valuation_requested` → **Numerical & Categorical inputs**
- `business_description` → **Text vectorized with TF-IDF**

Then fed into a pipeline:
```python
Pipeline([
  ("preprocessing", ColumnTransformer([...],
  ("model", XGBClassifier())
])


📂 Project Structure
├── app.py                     # Gradio frontend
├── ml_model_nlp.py            # NLP + ML model training script
├── shark_model_nlp.pkl        # Saved model
├── Cleaned_SharkTank.xlsx     # Cleaned dataset
├── shark_tank_us.csv          # Original dataset
├── requirements.txt           # Dependencies
├── check_col.py               # Helper script to check column names
├── Readme.md                  # This file


💻 How to Run Locally

Step 1: Clone the Repository
git clone https://github.com/krishanu2/shark-tank-predictor.git
cd shark-tank-predictor


📦 Step 2: Install Dependencies
python ml_model_nlp.py


🖥️ Step 4: Launch Gradio App
python app.py


🧪 Sample Prediction
Enter details like:

Industry: Food and Beverage

Ask Amount: 100000

Equity Offered: 10

Valuation Requested: 1000000

Description: We offer plant-based snacks for healthy lifestyle lovers...

✅ Output: Got a Deal

📈 Model Performance
Metric	Score
Accuracy	~58%
Precision	0.61
Recall	0.76
F1-Score	0.68

Model performance can be improved with more feature engineering, better text embeddings, or ensemble models.

✍️ Author
Krishanu Mahapatra

GitHub: @krishanu2

License
This project is open-source and available under the MIT License.


If you liked this project, leave a ⭐ on the repo to support more such projects!

