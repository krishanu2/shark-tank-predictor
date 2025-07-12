import gradio as gr
import joblib
import pandas as pd
import re
import string
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# Load model
model = joblib.load("shark_model_nlp.pkl")

# Text cleaner
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

# Prediction function
def predict_deal(industry, ask_amount, equity_offered, valuation_requested, business_description):
    # Create input DataFrame
    input_df = pd.DataFrame({
        'industry': [industry],
        'original_ask_amount': [float(ask_amount)],
        'original_offered_equity': [float(equity_offered)],
        'valuation_requested': [float(valuation_requested)],
        'ask_to_valuation_ratio': [float(ask_amount)/float(valuation_requested)],
        'business_description': [clean_text(business_description)]
    })

    prediction = model.predict(input_df)[0]
    return "✅ Deal" if prediction == 1 else "❌ No Deal"

# Gradio interface
industries = ['Technology', 'Food & Beverage', 'Fashion', 'Health & Wellness', 'Education', 'Home Goods', 'Entertainment', 'Others']  # Add more if needed

demo = gr.Interface(
    fn=predict_deal,
    inputs=[
        gr.Dropdown(industries, label="Industry"),
        gr.Number(label="Ask Amount (USD)"),
        gr.Number(label="Equity Offered (%)"),
        gr.Number(label="Valuation Requested (USD)"),
        gr.Textbox(label="Business Description", lines=4)
    ],
    outputs="text",
    title="Shark Tank Deal Predictor 🦈💰",
    description="Fill in startup details to predict if a deal will happen!"
)

demo.launch()
