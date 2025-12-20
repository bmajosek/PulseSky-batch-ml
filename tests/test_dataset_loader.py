"""Tests for dataset loader."""
from unittest.mock import patch, MagicMock
from src.dataset_loader import DatasetLoader


@patch("src.dataset_loader.col")
def test_load_posts(mock_col):
    mock_spark = MagicMock()
    mock_df = MagicMock()
    mock_selected = MagicMock()
    mock_filtered = MagicMock()
    
    mock_df.select.return_value = mock_selected
    mock_selected.filter.return_value = mock_filtered
    mock_filtered.filter.return_value = mock_filtered
    mock_spark.read.parquet.return_value = mock_df
    
    loader = DatasetLoader(mock_spark)
    result = loader.load_posts()
    
    assert mock_spark.read.parquet.called
    assert mock_df.select.called
    assert mock_selected.filter.called
