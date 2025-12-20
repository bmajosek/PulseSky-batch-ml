"""Tests for model wrapper."""
from unittest.mock import patch, MagicMock
from src.model_wrapper import ModelWrapper
import torch


@patch("src.model_wrapper.RobertaForSequenceClassification")
@patch("src.model_wrapper.RobertaTokenizer")
def test_model_wrapper_prediction(mock_tokenizer_class, mock_model_class):
    mock_tokenizer = MagicMock()
    token_output = {
        "input_ids": torch.tensor([[1, 2, 3]]),
        "attention_mask": torch.tensor([[1, 1, 1]])
    }
    mock_tokenizer.return_value = token_output
    mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
    
    mock_model = MagicMock()
    mock_model.logits = torch.tensor([[0.1, 0.2, 0.7]])
    mock_model_instance = MagicMock(return_value=mock_model)
    mock_model_instance.to.return_value = mock_model_instance
    mock_model_instance.eval.return_value = None
    mock_model_class.from_pretrained.return_value = mock_model_instance
    
    wrapper = ModelWrapper("fake_model_path")
    result = wrapper.predict_sentiment("This is great!")
    
    assert isinstance(result, str)
    assert result in ["negative", "neutral", "positive"]
