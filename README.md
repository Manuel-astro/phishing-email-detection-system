# Phishing Email Detection System

A machine learning-based cybersecurity project that identifies potentially malicious phishing emails using Natural Language Processing (NLP) and supervised machine learning.

## Project Overview

Phishing emails are commonly used by attackers to steal sensitive information, distribute malicious links, and compromise user accounts.

This project implements a phishing email detection system that analyzes email text and classifies messages as either legitimate (safe) or phishing.

The system uses TF-IDF feature extraction and Logistic Regression for classification, with a Streamlit dashboard providing an interactive interface.

## Features

- Email text classification.
- Natural Language Processing using TF-IDF.
- Logistic Regression machine learning model.
- Detection of safe and phishing emails.
- Interactive Streamlit dashboard.
- Model training and evaluation scripts.
- Prediction functionality for individual email messages.

## Technologies Used

- Python 3.13
- Scikit-learn
- Pandas
- NumPy
- TF-IDF Vectorization
- Logistic Regression
- Joblib
- Streamlit

## System Architecture

1. Email text is submitted through the dashboard.
2. The trained TF-IDF vectorizer transforms the text into numerical features.
3. The Logistic Regression classifier analyzes the extracted features.
4. The system predicts whether the email is safe or phishing.
5. The dashboard displays the prediction.

## Project Structure

```text
phishing-email-detection-system/
│
├── app.py
├── detector.py
├── predict.py
├── train_model.py
├── evaluate_model.py
├── phishing_model.joblib
├── requirements.txt
├── README.md
└── .gitignore
```

The trained model is included only if it is available in the repository.

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR-USERNAME/phishing-email-detection-system.git
cd phishing-email-detection-system
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
python -m streamlit run app.py
```

Open the local address displayed in your terminal, normally:

```text
http://localhost:8501
```

## Model Development

The model uses:

- TF-IDF with unigram and bigram features.
- Logistic Regression.
- Class balancing.
- Stratified train-test splitting.
- Classification metrics for model evaluation.

The initial internal test produced an accuracy of approximately 98.37% on a held-out split from the original dataset.

This result is an internal evaluation, not evidence of equivalent performance on unseen real-world email sources.

## Evaluation

The project includes an evaluation script for testing the trained model against a separate dataset.

Evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Independent evaluation and dataset overlap checks are ongoing.

## Limitations

- The current model primarily analyzes email text.
- It does not yet perform comprehensive URL reputation checks.
- It does not yet inspect full email headers.
- Performance may vary across different email datasets.
- A phishing prediction should not be treated as definitive proof that an email is malicious.

## Future Improvements

- URL and domain analysis.
- Email-header inspection.
- Detection of suspicious attachments.
- Evaluation using more diverse independent datasets.
- Improved model generalization.
- Additional machine learning approaches.

## Author

Manuel-astro

Cybersecurity | IT Security | Machine Learning

## Disclaimer

This project is developed for educational, research, and cybersecurity portfolio purposes. It is not a replacement for professional email security solutions.
