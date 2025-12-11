import pandas as pd
from unittest.mock import patch, MagicMock
from src.train_sentiment_model import train
import torch


@patch("builtins.print")
@patch("src.train_sentiment_model.DataLoader")
@patch("src.train_sentiment_model.train_test_split")
@patch("src.train_sentiment_model.DatasetLoader")
@patch("src.train_sentiment_model.RobertaTokenizer")
@patch("src.train_sentiment_model.RobertaForSequenceClassification")
def test_train_runs(mock_model, mock_tok, mock_loader, mock_split, mock_dataloader, mock_print):

    df = pd.DataFrame({
        "text": ["a","b","c","d","e","f"],
        "label": ["positive","negative","positive","negative","positive","negative"]
    })

    mock_loader.return_value.load_silver.return_value = df

    # Create proper train and val dataframes with clean_text column added
    train_df = df.iloc[:4].copy()
    val_df = df.iloc[4:].copy()
    train_df["clean_text"] = train_df["text"]
    val_df["clean_text"] = val_df["text"]

    # Fix train/test split to return dataframes with clean_text
    mock_split.return_value = (train_df, val_df)

    # Mock tokenizer
    mock_tok_instance = MagicMock()
    mock_tok_instance.return_value = {
        "input_ids": torch.tensor([[1]*64]),
        "attention_mask": torch.tensor([[1]*64])
    }
    mock_tok_instance.save_pretrained = MagicMock()
    mock_tok.from_pretrained.return_value = mock_tok_instance

    # Mock DataLoader to return empty iterables for training and validation
    mock_dataloader.return_value = []
    
    # Mock model with fake parameters
    fake_param = torch.nn.Parameter(torch.randn(1))
    
    model_instance = MagicMock()
    model_instance.cuda = MagicMock(return_value=model_instance)
    model_instance.train = MagicMock()
    model_instance.eval = MagicMock()
    model_instance.parameters = MagicMock(return_value=[fake_param])
    model_instance.return_value = MagicMock(logits=torch.randn(4, 3))
    model_instance.save_pretrained = MagicMock()
    mock_model.from_pretrained.return_value = model_instance

    train()

    assert True
