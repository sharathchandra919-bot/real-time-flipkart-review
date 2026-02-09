import os
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score


# load data
df = pd.read_csv("data/flipkart_reviews.csv")

df = df.dropna(subset=["Review text"])
X = df["Review text"].astype(str)

# Convert rating into sentiment label
y = df["Ratings"].apply(lambda r: 1 if r >= 4 else 0)

# split data
X_train_text, X_test_text, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# vectorizer
tfidf = TfidfVectorizer(max_features=5000)
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

# create folder
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)


mlflow.set_experiment("Flipkart_Sentiment_Analysis")

with mlflow.start_run(run_name="LR_TFIDF_maxfeat5000"):

    # log parameters
    mlflow.log_param("vectorizer", "tfidf")
    mlflow.log_param("model", "logistic_regression")
    mlflow.log_param("max_features", 9000)
    mlflow.log_param("random_state", 42)

    mlflow.set_tag("project", "flipkart_sentiment")
    mlflow.set_tag("embedding", "tfidf")
    mlflow.set_tag("algo", "logreg")


    # train model
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)

    # metrics
    f1 = f1_score(y_test, preds)
    mlflow.log_metric("f1_score", f1)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # classification report artifact
    report = classification_report(y_test, preds)
    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    mlflow.log_artifact("reports/classification_report.txt")

    # confusion matrix artifact
    cm = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("reports/confusion_matrix.png")
    plt.close()

    mlflow.log_artifact("reports/confusion_matrix.png")

    # save model + vectorizer
    joblib.dump(model, "models/logreg.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

    mlflow.log_artifact("models/logreg.pkl")
    mlflow.log_artifact("models/vectorizer.pkl")

    # log model properly in MLflow
    # mlflow.sklearn.log_model(model, "sentiment_model")
    mlflow.sklearn.log_model(
    model,
    name="sentiment_model",
    registered_model_name="FlipkartSentimentModel"
    # stage=staging
)
