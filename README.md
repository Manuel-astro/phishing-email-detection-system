# Phishing Email Detection System

A machine learning-based cybersecurity project that detects potentially malicious phishing emails using Natural Language Processing (NLP) and supervised machine learning.

## Project Overview

Phishing attacks are commonly used to steal sensitive information, distribute malicious links, and compromise user accounts.

This project implements a phishing email detection system that analyzes email text and classifies messages into two categories: **Safe** and **Phishing**.

The system uses TF-IDF (Term Frequency-Inverse Document Frequency) for text feature extraction and Logistic Regression for classification. A Streamlit dashboard provides an interactive interface for submitting and analysing email messages.

## Features

- Email text classification.
- Natural Language Processing using TF-IDF.
- Logistic Regression classification model.
- Detection of safe and phishing emails.
- Interactive Streamlit dashboard.
- Individual email prediction functionality.
- Model training and evaluation scripts.

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

The detection process follows these steps:

1. The user submits email text through the Streamlit dashboard.
2. The trained TF-IDF vectorizer transforms the email text into numerical features.
3. The Logistic Regression classifier analyses the extracted features.
4. The system predicts whether the email is safe or phishing.
5. The dashboard displays the classification result.

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

The repository includes the trained machine learning model. Dataset files and the Python virtual environment are excluded.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Manuel-astro/phishing-email-detection-system.git
cd phishing-email-detection-system
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

### 3. Activate the virtual environment

For Windows:

```bash
.venv\Scripts\activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
python -m streamlit run app.py
```

Open the local address displayed in your terminal. The default address is:

```text
http://localhost:8501
```

## Model Development

The machine learning pipeline uses:

- TF-IDF with unigram and bigram features.
- Logistic Regression classification.
- Class balancing.
- Stratified train-test splitting.
- Classification metrics for performance evaluation.

The initial internal evaluation produced an accuracy of approximately **98.37%** on a held-out test set from the original dataset.

This result represents internal test performance and should not be interpreted as proof of equivalent accuracy on independent real-world email sources.

## Evaluation

The project includes an evaluation script for assessing the trained model.

The evaluation process considers:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Independent validation and dataset overlap checks remain areas for further investigation.

## Limitations

- The current model primarily analyses email text.
- Comprehensive URL reputation checks are not yet implemented.
- Full email-header inspection is not yet supported.
- Suspicious attachment analysis is not yet implemented.
- Performance may vary across different email datasets.
- A phishing prediction should not be treated as definitive proof that an email is malicious.

## Future Improvements

- URL and domain reputation analysis.
- Email-header inspection.
- Suspicious attachment detection.
- Evaluation using more diverse independent datasets.
- Improved model generalisation.
- Exploration of additional machine learning algorithms.

## Author

**Manuel Quaye**

Cybersecurity | IT Security | Machine Learning

GitHub: [@Manuel-astro](https://github.com/Manuel-astro)

## Disclaimer

This project is developed for educational, research, and cybersecurity portfolio purposes. It is not a replacement for professional email security solutions.