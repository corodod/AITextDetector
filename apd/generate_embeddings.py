import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from model_utils import mean_pooling
import os

DATA_SPLITS = {
    "train": "RuATD/train.csv",
    "val": "RuATD/val.csv"
}

# Загрузка модели
print("[INFO] Загрузка модели эмбеддингов...")
tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/rubert-base-cased")
model = AutoModel.from_pretrained("DeepPavlov/rubert-base-cased")
# tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
# encoder = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru")

batch_size = 16
os.makedirs("data", exist_ok=True)

for split_name, csv_path in DATA_SPLITS.items():
    print(f"\n[INFO] Обработка {split_name} данных...")

    df = pd.read_csv(csv_path)
    df = df.rename(columns={col: col.strip().lower() for col in df.columns})

    if "class" not in df.columns or "text" not in df.columns:
        print(f"[ERROR] В {csv_path} отсутствует нужный столбец.")
        continue

    texts = df["text"].tolist()
    labels = torch.tensor([0 if lbl == "Human" else 1 for lbl in df["class"]])
    # Вычисление эмбеддингов
    all_embeddings = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
            model_output = model(**encoded)
            pooled = mean_pooling(model_output, encoded['attention_mask'])
            all_embeddings.append(pooled)
            print(f"[{split_name}] Обработано {min(i + batch_size, len(texts))}/{len(texts)}")

    sentence_embeddings = torch.cat(all_embeddings, dim=0)
    torch.save({
        "embeddings": sentence_embeddings,
        "labels": labels
    }, f"data/{split_name}_embeddings.pt")

    print(f"[INFO] Сохранено в data/{split_name}_embeddings.pt")
