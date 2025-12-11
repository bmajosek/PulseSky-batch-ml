import pandas as pd
import torch
from unittest.mock import MagicMock
from src.utils import SentimentDataset, preprocess_text


def test_preprocess_text():
    assert preprocess_text("Hello WORLD!!!") == "hello world"


def test_dataset():

    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1]*64]),
        "attention_mask": torch.tensor([[1]*64])
    }

    df = pd.DataFrame({
        "clean_text": ["good", "bad"],
        "label": ["positive", "negative"]
    })

    ds = SentimentDataset(df, tokenizer)

    item = ds[0]

    assert item["labels"] == 2  # "positive" maps to 2
    assert isinstance(item["input_ids"], torch.Tensor)
    assert isinstance(item["attention_mask"], torch.Tensor)
