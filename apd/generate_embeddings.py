# generate_embeddings.py

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from model_utils import mean_pooling

# Загрузка данных
df = pd.read_csv("RuATD/human_baseline.csv")
texts = df["text"].tolist()
label_map = {"H": 0, "M": 1}
labels = torch.tensor(df["majority_vote"].map(label_map).tolist())

# Загрузка модели
print("[INFO] Загрузка модели эмбеддингов...")
tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
encoder = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

# Вычисление эмбеддингов
print("[INFO] Вычисление эмбеддингов...")
batch_size = 16
all_embeddings = []

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
        model_output = encoder(**encoded)
        pooled = mean_pooling(model_output, encoded['attention_mask'])
        all_embeddings.append(pooled)
        print(f"[INFO] Обработано {min(i+batch_size, len(texts))}/{len(texts)}")

sentence_embeddings = torch.cat(all_embeddings, dim=0)
print(f"[INFO] Эмбеддинги сформированы: {sentence_embeddings.shape}")

# Сохранение
torch.save({
    "embeddings": sentence_embeddings,
    "labels": labels
}, "data/embeddings.pt")
print("[INFO] Сохранено в data/embeddings.pt")

# train.py

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.metrics import classification_report, accuracy_score
from model_utils import DeepMLPClassifier
from collections import Counter

# Загрузка эмбеддингов
print("[INFO] Загрузка подготовленных эмбеддингов...")
data = torch.load("data/embeddings.pt")
embeddings, labels = data["embeddings"], data["labels"]
print(f"[INFO] Загружено: embeddings {embeddings.shape}, labels {labels.shape}")

# Dataset
dataset = TensorDataset(embeddings, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)
print(f"[INFO] Размер train: {len(train_ds)}, val: {len(val_ds)}")

# Классификатор
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_mlp = DeepMLPClassifier(input_dim=embeddings.shape[1]).to(device)
optimizer = torch.optim.Adam(model_mlp.parameters(), lr=1e-3)

# Взвешивание классов
label_counts = Counter(labels.tolist())
total = sum(label_counts.values())
class_weights = [total / label_counts[0], total / label_counts[1]]
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Обучение
print("[INFO] Начало обучения...")
num_epochs = 20
for epoch in range(num_epochs):
    start_time = time.time()
    model_mlp.train()
    total_loss = 0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        output = model_mlp(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)

    # Валидация
    model_mlp.eval()
    val_loss = 0
    all_preds, all_true = [], []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model_mlp(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_true.extend(yb.cpu().tolist())

    avg_val_loss = val_loss / len(val_loader)
    val_acc = accuracy_score(all_true, all_preds)
    elapsed = time.time() - start_time

    print(f"[EPOCH {epoch+1}/{num_epochs}] "
          f"Train loss: {avg_train_loss:.4f} | "
          f"Val loss: {avg_val_loss:.4f} | "
          f"Val acc: {val_acc:.4f} | "
          f"Time: {elapsed:.1f} sec")

# Сохранение модели
torch.save(model_mlp.state_dict(), "saved_model.pt")
print("[INFO] Модель сохранена в saved_model.pt")

# Итоговая оценка
print("\n[INFO] Итоговая оценка модели:")
print(classification_report(all_true, all_preds, target_names=["Human", "AI"]))
