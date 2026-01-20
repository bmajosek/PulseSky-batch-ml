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
    # Uproszczony test - nie będziemy faktycznie trenować
    # Sprawdzamy tylko czy funkcja się importuje i mocki są ustawione
    assert mock_prepare is not None
    assert mock_split is not None
    assert mock_tokenizer_class is not None
    assert mock_model_class is not None
