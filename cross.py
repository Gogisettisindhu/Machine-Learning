import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Sample dataset
data = {
    "review": [
        "Great product, fast delivery and works perfectly!",
        "Worst product ever. Completely stopped working in a day.",
        "I liked the design, but performance could be better.",
        "Awesome value for money. Highly recommend this!",
        "Fake product! Don't buy!",
        "Very satisfied with the quality and service.",
        "Not worth the price. Battery dies quickly.",
        "Excellent build quality. Fast charging as advertised.",
        "Terrible experience. Arrived broken and late.",
        "Happy with the purchase. Will buy again."
    ],
    "label": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1]  # 1 = genuine, 0 = fake
}
df = pd.DataFrame(data)

# ------------------- Machine Learning Model -------------------
X_train, X_test, y_train, y_test = train_test_split(df["review"], df["label"], test_size=0.2, random_state=42)
tfidf = TfidfVectorizer(max_features=1000)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

ml_model = LogisticRegression()
ml_model.fit(X_train_vec, y_train)
ml_preds = ml_model.predict(X_test_vec)
print("=== ML Classification Report ===")
print(classification_report(y_test, ml_preds))

# ------------------- Deep Learning with BERT -------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encodings = tokenizer(df['review'].tolist(), truncation=True, padding=True, return_tensors='pt')
labels = torch.tensor(df['label'].tolist())

class ReviewDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.labels[idx]
        }

    def __len__(self):
        return len(self.labels)

dataset = ReviewDataset(encodings, labels)
train_size = int(0.8 * len(dataset))
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, len(dataset) - train_size])

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# trainer.train()  # Uncomment to start training
