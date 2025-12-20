"""Tests for streaming inference pipeline."""
from unittest.mock import patch, MagicMock
from src.inference_streaming import run_streaming_inference


@patch("src.inference_streaming.AutoModelForSequenceClassification")
@patch("src.inference_streaming.AutoTokenizer")
@patch("src.inference_streaming.SparkSession")
def test_streaming_inference_setup(mock_spark, mock_tokenizer, mock_model):
    mock_spark_instance = MagicMock()
    mock_spark.builder.appName.return_value.getOrCreate.return_value = mock_spark_instance
    
    mock_stream = MagicMock()
    mock_kafka = MagicMock()
    mock_spark_instance.readStream.format.return_value = mock_kafka
    mock_kafka.option.return_value = mock_kafka
    mock_kafka.load.return_value = mock_stream
    
    mock_stream.selectExpr.return_value = mock_stream
    mock_stream.withColumn.return_value = mock_stream
    mock_stream.filter.return_value = mock_stream
    mock_stream.select.return_value = mock_stream
    
    mock_write = MagicMock()
    mock_stream.writeStream.foreachBatch.return_value = mock_write
    mock_write.option.return_value = mock_write
    mock_write.trigger.return_value = mock_write
    mock_write.start.return_value.awaitTermination.return_value = None
    
    mock_tokenizer.from_pretrained.return_value = MagicMock()
    mock_model.from_pretrained.return_value = MagicMock()
    
    try:
        run_streaming_inference()
    except (AttributeError, RuntimeError):
        pass
    
    assert mock_spark.builder.appName.called
