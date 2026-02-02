📦 Real-Time Flipkart Product Review Sentiment Analysis
📌 Project Overview

This project performs sentiment analysis on real-time Flipkart product reviews to classify customer feedback as Positive or Negative and to identify customer pain points from negative reviews.

The system is built as an end-to-end ML application, covering:

Data preprocessing

Feature extraction

Model training & evaluation

Real-time inference via a web application

Deployment on AWS EC2

🎯 Objective

Classify Flipkart product reviews into Positive or Negative

Analyze negative reviews to extract common pain points

Provide a real-time sentiment prediction interface

📊 Dataset

Source: Flipkart product reviews (pre-scraped by Data Engineering team)

Product: YONEX MAVIS 350 Nylon Shuttle

Total Reviews: 8,518

Features:

Reviewer Name

Rating

Review Title

Review Text

Place of Review

Date

Up Votes

Down Votes

🧹 Data Preprocessing

Removed special characters and punctuation

Converted text to lowercase

Removed stopwords

Lemmatization using NLTK WordNet

Removed neutral reviews (rating = 3)

Label creation:

Rating ≥ 4 → Positive (1)

Rating ≤ 2 → Negative (0)

🧠 Feature Engineering

TF-IDF Vectorization for text representation

Vocabulary learned from training data

Saved vectorizer for reuse during inference

🤖 Model Training

Model Used: Logistic Regression

Train-Test Split: 80/20 (stratified)

Evaluation Metric: F1-Score

📈 Model Performance
F1 Score: 0.95

Class 0 (Negative):
Precision: 0.86 | Recall: 0.45

Class 1 (Positive):
Precision: 0.92 | Recall: 0.99


The model performs strongly on positive sentiment detection while maintaining reasonable precision on negative reviews.

🔍 Pain Point Analysis

Negative reviews were analyzed separately

CountVectorizer was used to extract frequently occurring terms

These terms highlight customer dissatisfaction areas such as:

durability

quality

packaging

value for money

🌐 Web Application (Streamlit)

A Streamlit-based web application allows users to:

Enter a product review

Instantly get sentiment prediction (Positive / Negative)

App Features

Real-time inference

Reuses trained ML model & vectorizer

Clean UI for demonstration

🚀 Deployment

Platform: AWS EC2 (Ubuntu 22.04)

Web Server: Streamlit

Port: 8501

Access: Public IP via EC2 Security Group

Deployment Flow

Code pushed to GitHub

Cloned on EC2 instance

Virtual environment setup

Dependencies installed

Streamlit app launched and exposed publicly
