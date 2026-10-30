
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

DATASET_PATH = "Phishing_Email.csv"
MODEL_PATH = "phishing_model.joblib"

print("Loading email dataset...")

df = pd.read_csv(DATASET_PATH)

# Keep the email text and its original label.
df = df[["Email Text", "Email Type"]].copy()
df = df.dropna(subset=["Email Text", "Email Type"])

df["Email Text"] = df["Email Text"].astype(str).str.strip()
df["Email Type"] = df["Email Type"].astype(str).str.strip()

df = df[df["Email Text"] != ""]

# Convert dataset labels to consistent names.
label_map = {
    "Safe Email": "safe",
    "Phishing Email": "phishing",
}

df["label"] = df["Email Type"].map(label_map)
df = df.dropna(subset=["label"])

# Remove exact duplicate email texts before splitting.
df = df.drop_duplicates(subset=["Email Text"])

print(f"Usable unique emails: {len(df)}")
print("\nClass distribution:")
print(df["label"].value_counts())

X = df["Email Text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

print(f"\nTraining examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=100000,
            sublinear_tf=True,
        ),
    ),
    (
        "classifier",
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        ),
    ),
])

print("\nTraining model. Please wait...")

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\n===== MODEL EVALUATION =====")
print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")

print("\nClassification report:")
print(
    classification_report(
        y_test,
        predictions,
        labels=["safe", "phishing"],
        target_names=["Safe Email", "Phishing Email"],
        zero_division=0,
    )
)

print("Confusion matrix (rows = actual, columns = predicted):")
print("Order: Safe Email, Phishing Email")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["safe", "phishing"],
))

joblib.dump(model, MODEL_PATH)

print(f"\nModel saved successfully to {MODEL_PATH}")
print("Training and evaluation complete.")
