import torch
import pandas as pd
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
import os

class TrafficNLP:
    def __init__(self, model_name='prajjwal1/bert-tiny', num_labels=4, save_dir='saved_nlp_model'):
        self.save_dir = save_dir
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        if os.path.exists(save_dir) and os.path.exists(os.path.join(save_dir, "model.safetensors")):
            print(f" Loading pre-trained NLP model from {save_dir}...")
            self.model = BertForSequenceClassification.from_pretrained(save_dir)
        else:
            print(f"Initializing base {model_name} model...")
            self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.label_map = {0: "Normal", 1: "Accident", 2: "Roadwork", 3: "Weather"}

    def prepare_data(self, csv_file):
        df = pd.read_csv(csv_file)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
        encodings = self.tokenizer(texts, truncation=True, padding=True, max_length=64, return_tensors='pt')
        dataset = TensorDataset(
            encodings['input_ids'], 
            encodings['attention_mask'], 
            torch.tensor(labels)
        )
        return dataset

    def train(self, csv_file, epochs=3, batch_size=8):
        print("Starting training (fine-tuning)...")
        dataset = self.prepare_data(csv_file)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=1e-5)
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                optimizer.zero_grad()
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")
        print("Training complete!")
        self.model.save_pretrained(self.save_dir)
        print(f" Model saved to '{self.save_dir}'")

    def predict(self, text):
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=64).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probs, dim=-1).item()
        return self.label_map[prediction], probs[0][prediction].item()

if __name__ == "__main__":
    data_file = "social_media_traffic.csv"
    if not os.path.exists(data_file):
        from nlp_data_generator import generate_nlp_data
        generate_nlp_data(data_file)
    nlp_system = TrafficNLP()
    nlp_system.train(data_file, epochs=3)
    test_texts = [
        "Major crash on the highway, avoid the area!",
        "Road construction is creating a lot of traffic today.",
        "Beautiful weather and the roads are clear.",
        "Heavy rain and flooded streets are slowing us down."
    ]
    print("\nTesting Model Predictions:")
    print("-" * 30)
    for text in test_texts:
        label, conf = nlp_system.predict(text)
        print(f"Tweet: \"{text}\"")
        print(f"Result: {label} (Confidence: {conf:.2%})\n")
