"""Model wrapper for RoBERTa sentiment classification."""
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

from src.utils import clean_text

LABELS = ["negative", "neutral", "positive"]


class ModelWrapper:
    def __init__(self, model_path: str):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    def predict_sentiment(self, text: str):
        if not text:
            return "neutral"

        text = clean_text(text)
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**inputs).logits
            idx = logits.argmax(dim=1).item()

        return LABELS[idx]
