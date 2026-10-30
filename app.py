
import joblib
import streamlit as st

MODEL_PATH = "phishing_model.joblib"

st.set_page_config(
    page_title="Phishing Email Detector",
    page_icon="🛡️",
    layout="centered",
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

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

email_text = st.text_area(
    "Paste email text",
    height=220,
    placeholder="Paste the email subject and body here...",
)

if st.button("Analyze Email", type="primary"):
    if not email_text.strip():
        st.warning("Please enter email text first.")
    else:
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
