import gradio as gr
import joblib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model
model = joblib.load("shark_model_nlp.pkl")

# Download stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

# Define industry options (add or change as needed)
INDUSTRY_OPTIONS = [
    "Food & Beverage", "Technology", "Health & Wellness", "Fashion", "Education",
    "Home & Kitchen", "Entertainment", "Sports", "Pets", "Beauty", "Other"
]

# Clean text
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

# Prediction function
def predict_deal(industry, ask_amount, equity_offered, valuation_requested, business_description):
    try:
        # Handle division by zero
        try:
            ask_to_valuation_ratio = float(ask_amount) / float(valuation_requested)
        except ZeroDivisionError:
            ask_to_valuation_ratio = 0.0

        cleaned_desc = clean_text(business_description)

        input_df = pd.DataFrame({
            'industry': [industry],
            'original_ask_amount': [float(ask_amount)],
            'original_offered_equity': [float(equity_offered)],
            'valuation_requested': [float(valuation_requested)],
            'ask_to_valuation_ratio': [ask_to_valuation_ratio],
            'business_description': [cleaned_desc]
        })

        prediction = model.predict(input_df)[0]
        return "✅ Deal" if prediction == 1 else "❌ No Deal"

    except Exception as e:
        print("❌ Error in prediction function:", str(e))
        return f"Error: {e}"

# Gradio UI
with gr.Blocks() as demo:
    gr.Markdown("# 🦈 Shark Tank Deal Predictor")

    with gr.Row():
        industry = gr.Dropdown(choices=INDUSTRY_OPTIONS, label="Industry", value="Technology")
        ask_amount = gr.Number(label="Ask Amount ($)", value=100000)
        equity_offered = gr.Number(label="Equity Offered (%)", value=10.0)
        valuation_requested = gr.Number(label="Valuation Requested ($)", value=1000000)

    business_description = gr.Textbox(label="Business Description", lines=5, placeholder="Describe the business...")

    submit = gr.Button("📈 Predict Deal")
    output = gr.Textbox(label="Prediction Result")

    submit.click(fn=predict_deal,
                 inputs=[industry, ask_amount, equity_offered, valuation_requested, business_description],
                 outputs=output)

# Launch app
if __name__ == "__main__":
    demo.launch()
