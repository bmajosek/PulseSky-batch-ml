# evaluate_model.py
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import torch

from src.dataset_loader import DatasetLoader
from src.utils import SentimentDataset, preprocess_text
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from src.config import cfg

def evaluate():
    df = DatasetLoader().load_silver(limit=2000)
    df["clean_text"] = df["text"].apply(preprocess_text)

    tok = RobertaTokenizer.from_pretrained(cfg.MODEL_PATH)
    ds = SentimentDataset(df, tok)
    dl = DataLoader(ds, batch_size=32)

    model = RobertaForSequenceClassification.from_pretrained(cfg.MODEL_PATH)
    model.cuda()
    model.eval()

    true, pred = [], []

    with torch.no_grad():
        for batch in dl:
            inputs = {k: v.cuda() for k, v in batch.items() if k != "labels"}
            logits = model(**inputs).logits
            p = logits.argmax(1).cpu().numpy()
            pred.extend(p)
            true.extend(batch["labels"].numpy())

    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average="macro")

    print("Accuracy:", acc)
    print("F1:", f1)

if __name__ == "__main__":
    evaluate()
