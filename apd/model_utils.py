import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from torch import nn

class DeepMLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden1=512, hidden2=128, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def load_all():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("ai-forever/sbert_large_nlu_ru")
    encoder_model = AutoModel.from_pretrained("ai-forever/sbert_large_nlu_ru").to(device)
    encoder_model.eval()

    # загрузка обученного MLP
    input_dim = 1024  # размер эмбеддинга sbert_large_nlu_ru
    mlp_model = DeepMLPClassifier(input_dim)
    mlp_model.load_state_dict(torch.load("saved_model.pt", map_location=device))
    mlp_model.to(device).eval()

    return tokenizer, encoder_model, mlp_model, device

def predict_text(text, tokenizer, encoder_model, mlp_model, device):
    encoded = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        model_output = encoder_model(**{k: v.to(device) for k, v in encoded.items()})
        embedding = mean_pooling(model_output, encoded["attention_mask"].to(device))
        logits = mlp_model(embedding)
        probs = torch.softmax(logits, dim=1)
    return probs[0, 1].item()  # вероятность класса "AI"
