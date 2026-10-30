
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# Load the labeled email dataset
data = pd.read_csv("emails.csv")

# Validate the dataset
required_columns = {"text", "label"}

if not required_columns.issubset(data.columns):
    raise ValueError("CSV must contain 'text' and 'label' columns.")

data = data.dropna(subset=["text", "label"])
data["text"] = data["text"].astype(str)
data["label"] = data["label"].astype(str).str.strip().str.lower()

allowed_labels = {"phishing", "legitimate"}

if not set(data["label"]).issubset(allowed_labels):
    raise ValueError("Labels must be phishing or legitimate.")

if data["label"].nunique() != 2:
    raise ValueError("Dataset must contain both classes.")

if data["label"].value_counts().min() < 2:
    raise ValueError("Each class needs at least two examples.")

# Split before fitting the model to avoid training on test examples
X_train, X_test, y_train, y_test = train_test_split(
    data["text"],
    data["label"],
    test_size=0.30,
    random_state=42,
    stratify=data["label"],
)

# Build the text-classification pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42,
    )),
])

# Train and evaluate
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\n=== Phishing Email Classifier ===")
print(f"Total emails: {len(data)}")
print(f"Training examples: {len(X_train)}")
print(f"Test examples: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, predictions):.2f}")

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=["legitimate", "phishing"],
    zero_division=0,
))

print("Confusion matrix (rows = actual, columns = predicted):")
print("Label order: legitimate, phishing")
print(confusion_matrix(
    y_test,
    predictions,
    labels=["legitimate", "phishing"],
))

# Save the complete pipeline
joblib.dump(model, "phishing_model.joblib")
print("\nModel saved as phishing_model.joblib")
