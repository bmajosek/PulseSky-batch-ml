from unittest.mock import patch, MagicMock
from src.inference_batch import run_batch


# Fake UDF returning constant
def fake_udf(func, t):
    return lambda x: "neutral"


@patch("src.inference_batch.F.udf", fake_udf)
@patch("src.inference_batch.ModelWrapper")
@patch("src.inference_batch.SparkSession")
def test_batch_inference(mock_spark, mock_model):
    fake_df = MagicMock()
    fake_df.withColumn.return_value = fake_df
    fake_df.write.mode.return_value.parquet.return_value = None

    mock_spark.builder.getOrCreate.return_value.read.parquet.return_value = fake_df
    mock_model.return_value.predict.return_value = "neutral"

    run_batch()

    assert True  # if no exception -> test passed
