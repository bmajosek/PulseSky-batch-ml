from unittest.mock import patch, MagicMock
from src.model_wrapper import ModelWrapper
import torch


@patch("src.model_wrapper.RobertaForSequenceClassification")
@patch("src.model_wrapper.RobertaTokenizer")
def test_model_wrapper_prediction(mock_tok, mock_model):

    # Mock tokenizer - wrap dict in object that has .to() method
    token_dict = {
        "input_ids": torch.tensor([[1, 2]]),
        "attention_mask": torch.tensor([[1, 1]])
    }
    
    class TokenOutput:
        def __init__(self, data):
            self.data = data
        def to(self, device):
            return self.data
    
    mock_tok_instance = MagicMock()
    mock_tok_instance.return_value = TokenOutput(token_dict)
    mock_tok.from_pretrained.return_value = mock_tok_instance

    # Mock model output
    logits = torch.tensor([[0.1, 0.2, 0.7]])
    
    model_instance = MagicMock()
    model_instance.eval = MagicMock()
    model_instance.cuda = MagicMock()
    model_instance.return_value = MagicMock(logits=logits)
    
    mock_model.from_pretrained.return_value = model_instance

    wrapper = ModelWrapper("fake_path")

    out = wrapper.predict("Hello world")

    assert "sentiment" in out
    assert out["confidence"] > 0
    assert out["sentiment"] == "positive"
