
from pathlib import Path

import joblib
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "phishing_model.joblib"
MAX_EMAIL_LENGTH = 50_000

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered",
)

@st.cache_resource
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError("Model file does not exist.")

    loaded_model = joblib.load(MODEL_PATH)

    if not hasattr(loaded_model, "predict_proba"):
        raise ValueError("The model does not support probability predictions.")

    if not hasattr(loaded_model, "classes_"):
        raise ValueError("The model does not expose its trained classes.")

    if "phishing" not in loaded_model.classes_:
        raise ValueError("The model does not contain a phishing class.")

    return loaded_model


st.title("🛡️ Phishing Email Detection System")

st.write(
    "Analyze email text for potential phishing. "
    "This is a research prototype, not a guarantee of safety."
)

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model not found. Run `python train_model.py` first.")
    st.stop()
except Exception as error:
    st.error(f"Unable to load the model: {error}")
    st.stop()

email_text = st.text_area(
    "Paste email text",
    height=220,
    max_chars=MAX_EMAIL_LENGTH,
    placeholder="Paste the email subject and body here...",
)

if st.button("Analyze Email", type="primary"):
    if not email_text.strip():
        st.warning("Please enter email text first.")

    else:
        try:
            prediction = model.predict([email_text])[0]
            probabilities = model.predict_proba([email_text])[0]

            classes = list(model.classes_)
            phishing_score = probabilities[classes.index("phishing")]

            st.subheader("Analysis Result")

            if prediction == "phishing":
                st.error("Potential phishing email")
            else:
                st.success("Classified as legitimate by the model")

            st.metric(
                "Model phishing score",
                f"{phishing_score:.1%}",
            )

            st.progress(float(phishing_score))

            st.caption(
                "The score is not a verified probability of malicious intent. "
                "A legitimate-looking result does not guarantee safety."
            )

        except Exception:
            st.error(
                "An error occurred while analyzing the email. "
                "Please check your input and try again."
            )
