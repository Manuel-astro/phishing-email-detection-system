
import re
from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_FILE = BASE_DIR / "phishing_model.joblib"
DATA_FILE = BASE_DIR / "meajor_validation.parquet.gzip"
TRAINING_FILE = BASE_DIR / "Phishing_Email.csv"
STARTER_FILE = BASE_DIR / "emails.csv"

OUTPUT_FILE = BASE_DIR / "meajor_misclassified_examples.csv"

BATCH_SIZE = 5000


def normalize_text(value):
    """Normalize text for exact-match overlap checks."""
    if pd.isna(value):
        return ""

    return re.sub(
        r"\s+",
        " ",
        str(value)
    ).strip().lower()


def main():

    if not MODEL_FILE.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}"
        )

    if not DATA_FILE.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {DATA_FILE}"
        )

    print("Loading existing model...")
    model = joblib.load(MODEL_FILE)

    if not hasattr(model, "predict"):
        raise ValueError("Loaded model does not support prediction.")

    print("Loading MeAJOR dataset...")
    data = pd.read_parquet(DATA_FILE)

    required = {"subject", "body", "label"}
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print("Original dataset rows:", len(data))

    data = data.dropna(subset=["label"]).copy()

    data["label"] = pd.to_numeric(
        data["label"],
        errors="coerce"
    )

    data = data[data["label"].isin([0, 1])].copy()

    if data.empty:
        raise ValueError("No valid labelled emails found.")

    # IMPORTANT:
    # Confirm the dataset documentation before relying
    # on this assumed label mapping.
    data["actual"] = data["label"].map({
        0: "safe",
        1: "phishing",
    })

    data["subject"] = (
        data["subject"].fillna("").astype(str).str.strip()
    )

    data["body"] = (
        data["body"].fillna("").astype(str).str.strip()
    )

    data["text"] = (
        data["subject"] + "\n\n" + data["body"]
    ).str.strip()

    data = data[data["text"].str.len() > 0].copy()

    data["_norm_text"] = data["text"].apply(normalize_text)
    data["_norm_body"] = data["body"].apply(normalize_text)

    before_dedup = len(data)

    data = data.drop_duplicates(
        subset="_norm_text"
    ).copy()

    print(
        "Duplicate messages removed:",
        before_dedup - len(data)
    )

    training_texts = set()

    if TRAINING_FILE.is_file():

        print("Checking overlap with original training dataset...")

        original = pd.read_csv(
            TRAINING_FILE,
            low_memory=False
        )

        if "Email Text" in original.columns:

            training_texts.update(
                original["Email Text"]
                .dropna()
                .apply(normalize_text)
            )

        else:
            print(
                "Warning: 'Email Text' column not found."
            )

    else:
        print(
            "WARNING: Original training CSV not found."
        )

    if STARTER_FILE.is_file():

        starter = pd.read_csv(STARTER_FILE)

        candidates = [
            column for column in starter.columns
            if column.strip().lower() in {
                "email text",
                "text",
                "email",
                "message",
                "content",
            }
        ]

        if candidates:

            training_texts.update(
                starter[candidates[0]]
                .dropna()
                .apply(normalize_text)
            )

    overlap = (
        data["_norm_text"].isin(training_texts)
        | data["_norm_body"].isin(training_texts)
    )

    overlap_count = int(overlap.sum())

    data = data.loc[~overlap].copy()

    print(
        "Exact training overlaps excluded:",
        overlap_count
    )

    print(
        "Unique emails available for evaluation:",
        len(data)
    )

    if data.empty:
        raise ValueError(
            "No emails remain after overlap removal."
        )

    print("\nActual label distribution:")
    print(data["actual"].value_counts().to_string())

    predictions = []

    print("\nPredicting emails in batches...")

    for start in range(0, len(data), BATCH_SIZE):

        batch = data["text"].iloc[
            start:start + BATCH_SIZE
        ]

        predictions.extend(model.predict(batch))

        print(
            f"Processed {min(start + BATCH_SIZE, len(data))}"
            f" / {len(data)} emails"
        )

    data["predicted"] = pd.Series(
        predictions,
        index=data.index
    )

    data["predicted"] = (
        data["predicted"].astype(str).str.lower()
    )

    unexpected = ~data["predicted"].isin(
        ["safe", "phishing"]
    )

    if unexpected.any():
        raise ValueError(
            "Unexpected model labels: "
            + str(
                data.loc[
                    unexpected,
                    "predicted"
                ].unique()
            )
        )

    y_true = data["actual"]
    y_pred = data["predicted"]

    print("\n========== MEAJOR EVALUATION ==========")

    print(f"Emails evaluated: {len(data)}")

    print(
        f"Accuracy: {accuracy_score(y_true, y_pred):.4f}"
    )

    print("\nClassification report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=["safe", "phishing"],
            target_names=[
                "Benign Email",
                "Phishing Email",
            ],
            zero_division=0,
        )
    )

    print(
        "Confusion matrix "
        "(rows = actual, columns = predicted)"
    )

    print("Label order: safe, phishing")

    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=["safe", "phishing"],
        )
    )

    if "source" in data.columns:

        print("\n========== RESULTS BY SOURCE ==========")

        for source, group in data.groupby(
            "source",
            dropna=False
        ):

            print(
                f"{source}: {len(group)} emails, "
                f"accuracy="
                f"{accuracy_score(group['actual'], group['predicted']):.4f}"
            )

    errors = data[
        data["actual"] != data["predicted"]
    ].copy()

    keep = [
        column for column in [
            "source",
            "subject",
            "body",
            "actual",
            "predicted",
        ]
        if column in errors.columns
    ]

    errors[keep].to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nMisclassified emails saved to:")
    print(OUTPUT_FILE)

    print(
        "\nEvaluation complete. "
        "The model was not retrained or overwritten."
    )


if __name__ == "__main__":
    main()
