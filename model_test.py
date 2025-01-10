import pandas as pd
import joblib
import torch
from transformers import (BertTokenizer, BertForSequenceClassification, 
                          DistilBertTokenizer, DistilBertForSequenceClassification, 
                          pipeline)

# Load the saved models and tokenizers
logistic_model = joblib.load('logistic_regression_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Load BERT model and tokenizer
bert_model = BertForSequenceClassification.from_pretrained('./bert_model')
bert_tokenizer = BertTokenizer.from_pretrained('./bert_model')

# Load DistilBERT model and tokenizer
distilbert_model = DistilBertForSequenceClassification.from_pretrained('./distilbert_model')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('./distilbert_model')

# Define a function to predict using Logistic Regression
def predict_logistic(new_tweets):
    new_tweets_tfidf = tfidf_vectorizer.transform(new_tweets)
    predictions = logistic_model.predict(new_tweets_tfidf)
    return predictions

# Define a function to predict using BERT
def predict_bert(new_tweets):
    bert_pipeline = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)
    predictions = [bert_pipeline(tweet)[0]['label'] for tweet in new_tweets]
    return predictions

# Define a function to predict using DistilBERT
def predict_distilbert(new_tweets):
    distilbert_pipeline = pipeline("text-classification", model=distilbert_model, tokenizer=distilbert_tokenizer)
    predictions = [distilbert_pipeline(tweet)[0]['label'] for tweet in new_tweets]
    return predictions

# Main function to take input and show results
def main():
    print("Enter tweets (type 'exit' to finish):")
    new_tweets = []
    
    while True:
        tweet = input("Tweet: ")
        if tweet.lower() == 'exit':
            break
        new_tweets.append(tweet)
    
    # Ensure we have input
    if not new_tweets:
        print("No tweets were entered. Exiting.")
        return

    # Get predictions from all models
    logistic_predictions = predict_logistic(new_tweets)
    bert_predictions = predict_bert(new_tweets)
    distilbert_predictions = predict_distilbert(new_tweets)

    # Combine results into a DataFrame
    results_df = pd.DataFrame({
        'Tweet': new_tweets,
        'Logistic Regression Prediction': logistic_predictions,
        'BERT Prediction': bert_predictions,
        'DistilBERT Prediction': distilbert_predictions
    })

    # Display the results
    print("\nResults from all models:")
    print(results_df)

    # Save results to a CSV file
    results_df.to_csv('model_predictions_results.csv', index=False)
    print("\nResults saved to 'model_predictions_results.csv'.")

if __name__ == "__main__":
    main()
