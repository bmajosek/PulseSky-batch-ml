"""Tests for batch inference pipeline."""
from unittest.mock import patch, MagicMock
import pandas as pd
from src.inference_batch import run_inference


@patch("src.inference_batch.prepare_pandas_dataset")
@patch("src.inference_batch.pipeline")
@patch("src.inference_batch.DatasetLoader")
@patch("src.inference_batch.SparkSession")
def test_batch_inference_runs(mock_spark, mock_loader, mock_pipeline, mock_prepare):
    mock_loader_instance = MagicMock()
    mock_df = MagicMock()
    mock_df.limit.return_value.count.return_value = 10
    mock_loader_instance.load_posts.return_value = mock_df
    mock_loader.return_value = mock_loader_instance
    
    pdf = pd.DataFrame({
        "text": ["Good product", "Bad service"],
        "created_at": ["2024-01-01", "2024-01-02"],
        "post_id": ["1", "2"],
        "language": ["en", "en"],
    })
    mock_df.toPandas.return_value = pdf
    mock_prepare.return_value = pdf
    
    mock_classifier = MagicMock()
    mock_classifier.return_value = [
        {"label": "positive", "score": 0.95},
        {"label": "negative", "score": 0.88},
    ]
    mock_pipeline.return_value = mock_classifier
    mock_spark.builder.appName.return_value.master.return_value.config.return_value = mock_spark.builder
    mock_spark_instance = MagicMock()
    mock_spark.builder.getOrCreate.return_value = mock_spark_instance
    
    run_inference()
    assert mock_loader_instance.load_posts.called
    assert mock_prepare.called
    assert mock_classifier.called
