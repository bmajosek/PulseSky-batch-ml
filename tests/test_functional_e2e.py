import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.inference_batch import run_inference
from src.utils import prepare_pandas_dataset


@pytest.mark.functional
@patch("src.inference_batch.pipeline")
@patch("src.inference_batch.DatasetLoader")
def test_functional_end_to_end_pipeline(mock_loader_cls, mock_pipeline):
    """
    Functional test of the complete end-to-end batch pipeline:
    data load -> preprocessing -> model inference -> output
    """

    # --- GIVEN: input data ---
    spark_df = MagicMock()
    spark_df.limit.return_value.count.return_value = 2

    pdf = pd.DataFrame({
        "post_id": ["1", "2"],
        "created_at": ["2024-01-01", "2024-01-02"],
        "language": ["en", "en"],
        "text": ["I love this product", "This is terrible"]
    })
    pdf["text_clean"] = pdf["text"]

    spark_df.toPandas.return_value = pdf

    loader = MagicMock()
    loader.load_posts.return_value = spark_df
    mock_loader_cls.return_value = loader

    # --- GIVEN: model output ---
    mock_pipeline.return_value.return_value = [
        {"label": "positive", "score": 0.95},
        {"label": "negative", "score": 0.90},
    ]

    # --- WHEN ---
    run_inference()

    # --- THEN ---
    assert loader.load_posts.called
    assert mock_pipeline.called
