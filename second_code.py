import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix
from transformers import (BertTokenizer, BertForSequenceClassification, Trainer, 
                          DistilBertTokenizer, DistilBertForSequenceClassification,
                          pipeline)
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib  # For loading the saved Logistic Regression model

# Step 2: Load the dataset
file_path = 'modified_labeled_data.csv'  # Path to the dataset
data = pd.read_csv(file_path)

# Step 3: Clean the column names by stripping whitespace
data.columns = data.columns.str.strip()

# Step 4: Define features (X) and labels (y)
X = data['tweet'].fillna('')  # Replace NaN values with an empty string
y = data['class']  # Labels

# Step 5: Split the data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load saved Logistic Regression model and vectorizer
log_model = joblib.load('logistic_regression_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Transform the test data using the saved TF-IDF vectorizer
X_test_tfidf = vectorizer.transform(X_test)

# ---------- Model 1: Logistic Regression ---------- #
print("Model 1: Logistic Regression with TF-IDF")
y_pred_log = log_model.predict(X_test_tfidf)
log_report = classification_report(y_test, y_pred_log, output_dict=True)
print("Logistic Regression Results:\n", log_report)

# ---------- Load Pretrained BERT Model ---------- #
print("\nModel 2: BERT Model")
# Load pretrained BERT model and tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('./bert_model')
bert_model = BertForSequenceClassification.from_pretrained('./bert_model')

# ---------- Load Pretrained DistilBERT Model ---------- #
print("\nModel 3: DistilBERT Model")
# Load pretrained DistilBERT model and tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('./distilbert_model')
distilbert_model = DistilBertForSequenceClassification.from_pretrained('./distilbert_model')

# Step 9: Create a Custom Dataset Class for PyTorch Models
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Encode labels
label_encoder = LabelEncoder()
y_test_encoded = label_encoder.fit_transform(y_test)

# Prepare test datasets for BERT and DistilBERT
test_dataset = TweetDataset(X_test.tolist(), y_test_encoded, bert_tokenizer, max_length=128)
test_dataset_distilbert = TweetDataset(X_test.tolist(), y_test_encoded, distilbert_tokenizer, max_length=128)

# ---------- ROC Curve Comparison (Fixed for BERT and DistilBERT) ---------- #
# Function to get logits (probability scores) for ROC curve from a Huggingface model
def get_model_logits(model, dataset):
    trainer = Trainer(model=model)
    predictions = trainer.predict(dataset).predictions
    logits = torch.tensor(predictions)
    probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()  # Taking probabilities for the positive class (class 1)
    return probabilities

# Get probability scores for ROC curve
y_prob_log = log_model.predict_proba(X_test_tfidf)[:, 1]  # For Logistic Regression
y_prob_bert = get_model_logits(bert_model, test_dataset)  # For BERT
y_prob_distilbert = get_model_logits(distilbert_model, test_dataset_distilbert)  # For DistilBERT

# Function to plot ROC Curve
def plot_roc_curve(y_true, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, label=f'{model_name} (area = {roc_auc:.2f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid()
    plt.savefig(f'roc_curve_{model_name}.png')
    plt.show()

# Plot ROC curves
plt.figure(figsize=(10, 6))
plot_roc_curve(y_test_encoded, y_prob_log, 'Logistic Regression')
plot_roc_curve(y_test_encoded, y_prob_bert, 'BERT')
plot_roc_curve(y_test_encoded, y_prob_distilbert, 'DistilBERT')

# ---------- Confusion Matrix ---------- #
# Function to get final predictions from a Huggingface model
def get_final_predictions(model, dataset):
    trainer = Trainer(model=model)
    predictions = trainer.predict(dataset).predictions
    pred_classes = np.argmax(predictions, axis=1)  # Get the class with the highest score
    return pred_classes

# Get final predictions for BERT and DistilBERT
y_pred_bert = get_final_predictions(bert_model, test_dataset)
y_pred_distilbert = get_final_predictions(distilbert_model, test_dataset_distilbert)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.savefig(f'confusion_matrix_{model_name}.png')
    plt.show()

# Plot confusion matrices
plot_confusion_matrix(y_test_encoded, y_pred_log, 'Logistic Regression')
plot_confusion_matrix(y_test_encoded, y_pred_bert, 'BERT')
plot_confusion_matrix(y_test_encoded, y_pred_distilbert, 'DistilBERT')

# ---------- Comparison Table ---------- #
comparison_data = {
    'Model': ['Logistic Regression', 'BERT', 'DistilBERT'],
    'Precision': [log_report['weighted avg']['precision'], 
                  classification_report(y_test_encoded, y_pred_bert, output_dict=True)['weighted avg']['precision'], 
                  classification_report(y_test_encoded, y_pred_distilbert, output_dict=True)['weighted avg']['precision']],
    'Recall': [log_report['weighted avg']['recall'], 
               classification_report(y_test_encoded, y_pred_bert, output_dict=True)['weighted avg']['recall'], 
               classification_report(y_test_encoded, y_pred_distilbert, output_dict=True)['weighted avg']['recall']],
    'F1 Score': [log_report['weighted avg']['f1-score'], 
                 classification_report(y_test_encoded, y_pred_bert, output_dict=True)['weighted avg']['f1-score'], 
                 classification_report(y_test_encoded, y_pred_distilbert, output_dict=True)['weighted avg']['f1-score']]
}

comparison_df = pd.DataFrame(comparison_data)
print("\nModel Comparison Table:")
print(comparison_df)

# Save comparison table to CSV
comparison_df.to_csv('model_comparison_table.csv', index=False)
