"""Tests for sentiment model training."""
import pandas as pd
from unittest.mock import patch, MagicMock
from src.train_sentiment_model import train_model
import torch


@patch("src.train_sentiment_model.Trainer")
@patch("src.train_sentiment_model.TrainingArguments")
@patch("src.train_sentiment_model.AutoModelForSequenceClassification")
@patch("src.train_sentiment_model.AutoTokenizer")
@patch("src.train_sentiment_model.train_test_split")
@patch("src.train_sentiment_model.prepare_pandas_dataset")
def test_train_model_runs(
    mock_prepare, 
    mock_split, 
    mock_tokenizer_class,
    mock_model_class,
    mock_args_class,
    mock_trainer_class
):
    pdf = pd.DataFrame({
        "text": ["good", "bad", "neutral", "excellent", "terrible", "okay"],
        "sentiment_label": ["positive", "negative", "neutral", "positive", "negative", "neutral"]
    })
    
    pdf["text_clean"] = pdf["text"]
    mock_prepare.return_value = pdf
    
    train_df = pdf.iloc[:4].copy()
    eval_df = pdf.iloc[4:].copy()
    mock_split.return_value = (train_df, eval_df)
    
    mock_tokenizer = MagicMock()
    mock_tokenizer.return_value = {
        "input_ids": torch.tensor([[1]*128]),
        "attention_mask": torch.tensor([[1]*128])
    }
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    
    mock_model = MagicMock()
    mock_model_class.from_pretrained.return_value = mock_model
    
    mock_trainer = MagicMock()
    mock_trainer_class.return_value = mock_trainer
    
    train_model(pdf)
    
    assert mock_prepare.called
    assert mock_split.called
    assert mock_trainer.train.called
    assert mock_trainer.save_model.called
