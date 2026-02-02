import pandas as pd
import joblib
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from preprocessing import clean_text

# Load trained artifacts
model = joblib.load("models/model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Reload data
df = pd.read_csv("data/flipkart_reviews.csv")

# Same labeling logic
df = df[df['Ratings'] != 3]
df['sentiment'] = df['Ratings'].apply(lambda x: 1 if x >= 4 else 0)

# Clean text
df['cleaned_review'] = df['Review text'].apply(clean_text)

# Vectorize
X = vectorizer.transform(df['cleaned_review'])
y = df['sentiment']

# Recreate test split (same seed)
_, X_test, _, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Evaluate
preds = model.predict(X_test)

print("F1 Score:", f1_score(y_test, preds))
print(classification_report(y_test, preds))
