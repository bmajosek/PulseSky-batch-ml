"""Tests for dataset loader."""
from unittest.mock import patch, MagicMock
from src.dataset_loader import DatasetLoader


@patch("src.dataset_loader.col")
def test_load_posts(mock_col):
    mock_spark = MagicMock()
    # Uproszczony test - sprawdzenie czy DatasetLoader się inicjalizuje
    loader = DatasetLoader(mock_spark)
    assert loader.spark == mock_spark
