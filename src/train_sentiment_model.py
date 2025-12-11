# train_model.py
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import train_test_split

from src.dataset_loader import DatasetLoader
from src.utils import SentimentDataset, preprocess_text
from src.config import cfg

def train():
    df = DatasetLoader().load_silver(limit=5000)
    df["clean_text"] = df["text"].apply(preprocess_text)

    train_df, val_df = train_test_split(df, test_size=0.1, stratify=df["label"])

    tok = RobertaTokenizer.from_pretrained("roberta-base")
    train_ds = SentimentDataset(train_df, tok)
    val_ds = SentimentDataset(val_df, tok)

    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=16)

    model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=3).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)

    for epoch in range(2):
        model.train()
        for batch in train_dl:
            batch = {k: v.cuda() for k, v in batch.items()}
            logits = model(**batch).logits
            loss = torch.nn.CrossEntropyLoss()(logits, batch["labels"])
            loss.backward()
            opt.step()
            opt.zero_grad()

        # simple validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.cuda() for k, v in batch.items()}
                pred = model(**batch).logits.argmax(1)
                correct += (pred == batch["labels"]).sum().item()
                total += len(pred)

        if total > 0:
            print("Epoch", epoch + 1, "Val Acc:", correct / total)

    model.save_pretrained(cfg.MODEL_PATH)
    tok.save_pretrained(cfg.MODEL_PATH)

if __name__ == "__main__":
    train()
