import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import pipeline
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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

# ---------- Model 1: Logistic Regression using TF-IDF ---------- #
print("Model 1: Logistic Regression with TF-IDF")
# Step 6: TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 7: Train Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_tfidf, y_train)

# Step 8: Predictions and Evaluation
y_pred_log = log_model.predict(X_test_tfidf)
log_report = classification_report(y_test, y_pred_log, output_dict=True)
print("Logistic Regression Results:\n", log_report)

# ---------- Model 2: BERT Model ---------- #
print("\nModel 2: BERT Model")
# Step 9: Create a Custom Dataset Class
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

# Tokenize the data using BERT's tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare datasets
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

train_dataset = TweetDataset(X_train.tolist(), y_train_encoded, bert_tokenizer, max_length=128)
test_dataset = TweetDataset(X_test.tolist(), y_test_encoded, bert_tokenizer, max_length=128)

# Load BERT model for sequence classification
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

# Train BERT model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=bert_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Fine-tune BERT
trainer.train()

# Predictions for BERT
bert_pipeline = pipeline("text-classification", model=bert_model, tokenizer=bert_tokenizer)

# Get predictions from BERT pipeline
y_pred_bert_raw = [bert_pipeline(t)[0]['label'] for t in X_test]

# Map 'LABEL_0', 'LABEL_1', etc., to their integer counterparts (0, 1, 2)
y_pred_bert = [int(label.split('_')[-1]) for label in y_pred_bert_raw]

# Ensure predictions are in numerical form and check if any unseen labels exist
y_pred_bert = np.array(y_pred_bert)

# Check for unseen labels
unseen_labels = set(y_pred_bert) - set(label_encoder.transform(label_encoder.classes_))

if unseen_labels:
    print(f"Warning: Found unseen labels in prediction: {unseen_labels}")
    # Map unseen labels to a default known class, such as 0
    y_pred_bert = [label if label in label_encoder.transform(label_encoder.classes_) else 0 for label in y_pred_bert]

# Now calculate the classification report
bert_report = classification_report(y_test_encoded, y_pred_bert, output_dict=True)
print("BERT Model Results:\n", bert_report)

# ---------- Model 3: DistilBERT Model ---------- #
print("\nModel 3: DistilBERT Model")
# Step 10: Tokenize using DistilBERT tokenizer
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Prepare DistilBERT datasets
train_dataset_distilbert = TweetDataset(X_train.tolist(), y_train_encoded, distilbert_tokenizer, max_length=128)
test_dataset_distilbert = TweetDataset(X_test.tolist(), y_test_encoded, distilbert_tokenizer, max_length=128)

# Load DistilBERT model for sequence classification
distilbert_model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=len(label_encoder.classes_))

# Train DistilBERT model
trainer_distilbert = Trainer(
    model=distilbert_model,
    args=training_args,
    train_dataset=train_dataset_distilbert,
    eval_dataset=test_dataset_distilbert
)

# Fine-tune DistilBERT
trainer_distilbert.train()

# Predictions for DistilBERT
distilbert_pipeline = pipeline("text-classification", model=distilbert_model, tokenizer=distilbert_tokenizer)
y_pred_distilbert_raw = [distilbert_pipeline(t)[0]['label'] for t in X_test]
y_pred_distilbert = [int(label.split('_')[-1]) for label in y_pred_distilbert_raw]

# Ensure predictions are in numerical form and check if any unseen labels exist
y_pred_distilbert = np.array(y_pred_distilbert)

# Check for unseen labels
unseen_labels = set(y_pred_distilbert) - set(label_encoder.transform(label_encoder.classes_))

if unseen_labels:
    print(f"Warning: Found unseen labels in prediction: {unseen_labels}")
    # Map unseen labels to a default known class, such as 0
    y_pred_distilbert = [label if label in label_encoder.transform(label_encoder.classes_) else 0 for label in y_pred_distilbert]

# Now calculate the classification report
distilbert_report = classification_report(y_test_encoded, y_pred_distilbert, output_dict=True)
print("DistilBERT Model Results:\n", distilbert_report)

# ---------- Graphical Representations ---------- #
# Prepare data for visualizations
model_names = ['Logistic Regression', 'BERT', 'DistilBERT']
precision = [log_report['weighted avg']['precision'], bert_report['weighted avg']['precision'], distilbert_report['weighted avg']['precision']]
recall = [log_report['weighted avg']['recall'], bert_report['weighted avg']['recall'], distilbert_report['weighted avg']['recall']]
f1_score = [log_report['weighted avg']['f1-score'], bert_report['weighted avg']['f1-score'], distilbert_report['weighted avg']['f1-score']]

# Bar Graph
plt.figure(figsize=(12, 6))
bar_width = 0.25
bar_positions = np.arange(len(model_names))

plt.bar(bar_positions, precision, bar_width, label='Precision', color='b')
plt.bar(bar_positions + bar_width, recall, bar_width, label='Recall', color='g')
plt.bar(bar_positions + 2 * bar_width, f1_score, bar_width, label='F1 Score', color='r')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Comparison')
plt.xticks(bar_positions + bar_width/2, model_names)  # Adjust xticks for centering
plt.legend()
plt.grid(axis='y')
plt.savefig('model_performance_bar_graph.png')
plt.show()

# Line Graph
plt.figure(figsize=(12, 6))
plt.plot(model_names, precision, marker='o', label='Precision')
plt.plot(model_names, recall, marker='o', label='Recall')
plt.plot(model_names, f1_score, marker='o', label='F1 Score')

plt.xlabel('Models')
plt.ylabel('Scores')
plt.title('Model Performance Line Graph')
plt.legend()
plt.grid()
plt.savefig('model_performance_line_graph.png')
plt.show()
