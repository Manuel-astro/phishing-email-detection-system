
from pathlib import Path

import joblib

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_model.joblib"


def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Model not found. Run: python train_model.py"
        )

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "predict"):
        raise ValueError("The loaded model cannot make predictions.")

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "The loaded model does not support probability predictions."
        )

    if not hasattr(model, "classes_"):
        raise ValueError("The model does not expose its trained classes.")

    if "phishing" not in model.classes_:
        raise ValueError("The model does not contain a phishing class.")

    return model


def main():
    try:
        model = load_model()

    except Exception as error:
        print(f"Unable to load model: {error}")
        return

    print("=== Machine Learning Email Detector ===")
    print("Paste your email below.")
    print("Type END on a new line when finished.")
    print("Type EXIT on the first line to quit.\n")

    while True:
        lines = []

        while True:
            line = input()

            if not lines and line.strip().upper() == "EXIT":
                print("Detector closed.")
                return

            if line.strip().upper() == "END":
                break

            lines.append(line)

        email_text = "\n".join(lines).strip()

        if not email_text:
            print("No email provided. Try again.\n")
            continue

        try:
            prediction = model.predict([email_text])[0]

            probabilities = model.predict_proba([email_text])[0]

            classes = list(model.classes_)

            phishing_probability = probabilities[
                classes.index("phishing")
            ]

            print(f"\nPrediction: {prediction.upper()}")

            print(
                "Model phishing score: "
                f"{phishing_probability:.1%}"
            )

            print(
                "This score is a model estimate, not a "
                "verified probability of malicious intent."
            )

            print(
                "Do not rely on this prototype alone to decide "
                "whether an email is safe."
            )

        except Exception as error:
            print(f"An error occurred during prediction: {error}")

        print("\nEnter another email, or type EXIT to quit.\n")


if __name__ == "__main__":
    main()
