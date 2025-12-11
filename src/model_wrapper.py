import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from src.utils import preprocess_text

LABELS = ["negative", "neutral", "positive"]

class ModelWrapper:
    def __init__(self, model_path="./models/sentiment_model"):
        self.tokenizer = RobertaTokenizer.from_pretrained(model_path)
        self.model = RobertaForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.model.cuda()

    def predict(self, text: str):
        if not text:
            return {"sentiment": "neutral", "confidence": 1.0}

        text = preprocess_text(text)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to("cuda")

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]

        idx = logits.argmax().item()
        return {
            "sentiment": LABELS[idx],
            "confidence": float(probs[idx])
        }
