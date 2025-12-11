import pandas as pd
from unittest.mock import patch, MagicMock
from src.dataset_loader import DatasetLoader


@patch("src.dataset_loader.SparkSession")
def test_load_silver(mock_spark):

    # Pandas DF zwracany przez toPandas()
    raw_pdf = pd.DataFrame({
        "text": ["good"],
        "sentiment_label": ["positive"]
    })

    # Spark DF chain
    spark_df = MagicMock()
    select_mock = MagicMock()
    dropna_mock = MagicMock()

    dropna_mock.toPandas.return_value = raw_pdf

    spark_df.select.return_value = select_mock
    select_mock.dropna.return_value = dropna_mock

    # Mock Spark Session
    mock_spark.builder.appName().getOrCreate().read.parquet.return_value = spark_df

    df = DatasetLoader().load_silver(limit=10)

    assert len(df) == 1
    assert df.iloc[0]["label"] == "positive"
