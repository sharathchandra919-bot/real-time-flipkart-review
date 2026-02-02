import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from preprocessing import clean_text
from feature_processing import get_tfidf_features

# Load data
df = pd.read_csv("data/flipkart_reviews.csv")

# Label creation
df = df[df['Ratings'] != 3]
df['sentiment'] = df['Ratings'].apply(lambda x: 1 if x >= 4 else 0)

# Clean text
df['cleaned_review'] = df['Review text'].apply(clean_text)

# TF-IDF
X, vectorizer = get_tfidf_features(df['cleaned_review'])
y = df['sentiment']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Save
joblib.dump(model, "models/model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model training completed and saved.")
