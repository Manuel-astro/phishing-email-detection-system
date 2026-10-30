
import joblib

MODEL_PATH = "phishing_model.joblib"


def main():
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print("Model not found. Run: python train_model.py")
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

        prediction = model.predict([email_text])[0]
        probabilities = model.predict_proba([email_text])[0]
        classes = model.classes_

        phishing_probability = probabilities[
            list(classes).index("phishing")
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
        print("\nEnter another email, or type EXIT to quit.\n")


if __name__ == "__main__":
    main()
