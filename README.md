# Sentiment Analysis with Logistic Regression + Streamlit

This project implements a **Sentiment Analysis Model** trained on Twitter and Reddit datasets using **Logistic Regression** and **TF-IDF Vectorizer**.  
The model is deployed as an interactive web application with **Streamlit**, where users can input text and receive instant predictions of sentiment: **Positive, Neutral, or Negative**.

---

## 🚀 Features
- Logistic Regression model with TF-IDF vectorization
- Handles Positive, Negative, and Neutral sentiment classification
- Interactive Streamlit app for real-time predictions
- Ready-to-use saved model and vectorizer (`.pkl` files)

---

## 🛠️ Tech Stack
- **Python 3.10+**
- **Scikit-learn** (Logistic Regression, TF-IDF)
- **Streamlit** (Web App Deployment)
- **NLTK** (Stopwords, preprocessing)
- **Pandas & NumPy**

---

## 📂 Project Structure

├── app.py # Streamlit app

├── LogReg_Sentiment_Model.pkl # Saved logistic regression model

├── Tfidf_Vectorizer.pkl # Saved TF-IDF vectorizer

├── DataSets

    ├── Reddit_Data.csv # Reddit dataset (preprocessed)
    ├── Twitter_Data.csv # Twitter dataset (preprocessed)
    
├── README.md # Project documentation

---

## 📊 Model Performance
**Accuracy:** 85.2%

**Classification Report:**
               precision    recall  f1-score   support

          -1       0.80      0.77      0.78      8674
           0       0.83      0.92      0.87     13636
           1       0.90      0.84      0.87     17644

    accuracy                           0.85     39954
    macro avg       0.84      0.84      0.84     39954
    weighted avg    0.85      0.85      0.85     39954

