import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Load dataset
df = pd.read_csv("shark_tank_us.csv")

# 2. Strip column names and fix names for consistency
df.columns = df.columns.str.strip()

# 3. Rename columns for uniform access
df.rename(columns={
    "Original Ask Amount": "Ask_Amount",
    "Original Offered Equity": "Equity_Offered"
}, inplace=True)

# 4. Check and drop missing required fields
required_cols = ['Industry', 'Ask_Amount', 'Equity_Offered', 'Valuation Requested', 'Business Description', 'Got Deal']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"❌ Missing columns: {missing}")

df = df.dropna(subset=required_cols)

# 5. Clean/convert fields
df["Got Deal"] = df["Got Deal"].astype(int)

# 6. Define features & target
X = df[["Industry", "Ask_Amount", "Equity_Offered", "Valuation Requested", "Business Description"]]
y = df["Got Deal"]

# 7. Preprocessing steps
numeric_features = ["Ask_Amount", "Equity_Offered", "Valuation Requested"]
numeric_transformer = StandardScaler()

categorical_features = ["Industry"]
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

text_features = "Business Description"
text_transformer = TfidfVectorizer(stop_words="english", max_features=50)

# 8. Combine transformations
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
        ("text", text_transformer, text_features)
    ]
)

# 9. Build pipeline
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

# 10. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 11. Fit the model
clf.fit(X_train, y_train)

# 12. Evaluate
y_pred = clf.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("📊 Classification Report:\n", classification_report(y_test, y_pred))

# 13. Save model
joblib.dump(clf, "shark_model.pkl")
print("📦 Model saved as shark_model.pkl")

# 14. Sample prediction
sample = pd.DataFrame([{
    "Industry": "Education",
    "Ask_Amount": 500000,
    "Equity_Offered": 15,
    "Valuation Requested": 5000000,
    "Business Description": "A VR platform to teach science to children in rural areas"
}])
sample_pred = clf.predict(sample)[0]
print(f"\n🧾 Sample Prediction: {'✅ Deal' if sample_pred == 1 else '❌ No Deal'}")
