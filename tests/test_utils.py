"""Tests for utility functions."""
import pandas as pd
from src.utils import clean_text, prepare_pandas_dataset


def test_clean_text():
    assert clean_text("Hello WORLD!!!") == "hello world!!!"
    assert clean_text("Check https://example.com now") == "check now"
    assert clean_text("Hi @user how are you") == "hi how are you"
    assert clean_text("hello    world") == "hello world"
    assert clean_text("") == ""
    assert clean_text(None) == ""


def test_prepare_pandas_dataset():
    df = pd.DataFrame({
        "text": [
            "   Hello WORLD   ",
            "https://example.com visit now",
            "hi",
            "Good product @user",
        ]
    })
    result = prepare_pandas_dataset(df)
    assert "text_clean" in result.columns
    assert len(result) == 3
    
    df2 = pd.DataFrame({
        "text_clean": ["Hello world", "short", "This is a good review"]
    })
    result2 = prepare_pandas_dataset(df2)
    assert "text_clean" in result2.columns
    assert len(result2) == 2
    
    df3 = pd.DataFrame({"other_col": ["test"]})
    try:
        prepare_pandas_dataset(df3)
        assert False, "Should raise KeyError"
    except KeyError:
        assert True
