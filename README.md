Sentiment Analysis of Amazon Reviews using Machine Learning and Transformers
📌 Project Overview
This project investigates the effectiveness of classical machine learning models and transformer-based deep learning models for sentiment analysis of Amazon product reviews. The goal is to automatically classify customer reviews into positive, neutral, and negative sentiment categories using Natural Language Processing (NLP) techniques.

The project compares Logistic Regression, Random Forest, and DistilBERT under a consistent experimental framework, analysing their performance, generalisation ability, and practical applicability on real-world, noisy, and imbalanced textual data.

This repository contains the full pipeline including data preprocessing, exploratory data analysis (EDA), model training, evaluation, and analysis.**

📂 Dataset

Source: Amazon Reviews Dataset (accessed via Kaggle)

Original Creators: Julian McAuley & Jure Leskovec (UC San Diego)

Type: User-generated textual data with metadata

Labels: Sentiment derived from star ratings

1–2 → Negative

3 → Neutral

4–5 → Positive

Key challenges in the dataset include:

Class imbalance

Noisy and informal language

Varying review lengths

Duplicate and missing values

🧹 Data Preprocessing

The following preprocessing steps were applied:

Removal of duplicates and missing reviews

Text cleaning (lowercasing, punctuation removal, stopwords removal)

Lemmatization

TF-IDF vectorization (for classical models)

Tokenization using DistilBERT tokenizer (for transformer model)

Handling class imbalance using oversampling and class-weighted loss

🤖 Models Implemented

Three models representing different learning paradigms were evaluated:

Logistic Regression

TF-IDF features

Strong baseline model

Interpretable and computationally efficient

Random Forest

Ensemble learning approach

Captures non-linear relationships

Prone to overfitting on sparse text features

DistilBERT

Transformer-based deep learning model

Captures contextual and semantic meaning

Best overall performance but sensitive to overfitting

Hyperparameter tuning was performed for Logistic Regression and Random Forest using cross-validation.

📈 Evaluation Metrics

Models were evaluated using:

Accuracy

Precision

Recall

F1-score (macro and weighted)

Confusion matrices

Learning curves (for overfitting analysis)

Training vs validation loss (for DistilBERT)

The weighted F1-score was prioritised due to class imbalance.

🏆 Key Results

DistilBERT achieved the highest overall performance.

Logistic Regression provided strong baseline results with low computational cost.

Random Forest showed overfitting and did not outperform simpler models.

Neutral sentiment remained difficult to classify across all models due to limited samples.

🧠 Key Findings

Contextual models outperform frequency-based models on sentiment tasks.

Increased model complexity does not guarantee better generalisation.

Class imbalance is a major limiting factor in sentiment classification.

Transformer models require careful regularisation and sufficient data.

🛠️ Technologies Used

Python

Pandas, NumPy

Scikit-learn

NLTK

Imbalanced-learn

Hugging Face Transformers

PyTorch

Matplotlib, Seaborn

🚀 Applications

This work can be applied to:

Customer feedback monitoring

Product quality analysis

Reputation management

E-commerce recommendation systems

Business intelligence dashboards

🔮 Future Work

Address class imbalance using advanced techniques (e.g. focal loss, data augmentation)

Aspect-based sentiment analysis

Multilingual sentiment classification

Larger-scale fine-tuning of transformer models

Real-time sentiment analysis pipelines

👤 Author

Muhammad Saad Bin Sagheer
MSc Data Science
University of Hertfordshire



