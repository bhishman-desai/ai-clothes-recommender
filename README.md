# H&M Fashion Recommender System

This project implements a personalized fashion recommendation system based on the [H&M Personalized Fashion Recommendations](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations) Kaggle competition.

## Overview

The system recommends clothing items to customers based on their purchase history, demographics, and preferences using advanced NLP and similarity search techniques.

### Features

- Customer profile analysis including:
  - Purchase history aggregation
  - Preferred colors and garment types
  - Club membership status
  - Age-based preferences

- Article (clothing item) analysis including:
  - Color groups
  - Product types
  - Garment categories
  - Detailed descriptions

- Machine Learning Components:
  - Text embeddings using SentenceTransformer (all-MiniLM-L6-v2)
  - Fast similarity search using FAISS
  - Customer-Article matching based on embedded features

## Technical Implementation

1. **Data Preprocessing**
   - Merges transaction data with article details
   - Aggregates customer purchase history
   - Handles missing values in customer data

2. **Feature Engineering**
   - Generates textual descriptions for customers and articles
   - Creates meaningful representations of customer preferences
   - Combines multiple features into rich text descriptions

3. **Model Architecture**
   - Uses SentenceTransformer for text embedding
   - Implements FAISS index for efficient similarity search
   - Provides top-k recommendations for each customer

## Dataset Structure

The project uses the following data files:
- articles.csv: Clothing item details
- customers.csv: Customer information
- transactions_train.csv: Historical purchase data

## Requirements

- Python packages:
  - pandas
  - sentence-transformers
  - faiss-cpu
  - numpy

## Usage

The main functionality is implemented in `index.ipynb`. To get recommendations for a specific customer:

```python
recommendations = recommend_clothes(customer_id, top_k=5)
```

This will return the top 5 recommended articles based on the customer's profile and preferences.

## Project Structure

```
clothes-predictor/
├── index.ipynb          # Main implementation notebook
├── dataset/
│   ├── articles.csv
│   ├── customers.csv
│   ├── transactions_train.csv
│   └── sample_submission.csv
└── README.md
```
