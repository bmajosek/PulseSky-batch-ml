from unittest.mock import patch, MagicMock
from src.inference_streaming import run_stream


class FakeCol:
    def __init__(self, name):
        self.name = name

    def cast(self, *_):
        return self


def fake_udf(func, t):
    return lambda x: "neutral"


@patch("src.inference_streaming.F.col", lambda x: FakeCol(x))
@patch("src.inference_streaming.F.from_json", lambda *a, **k: MagicMock())
@patch("src.inference_streaming.F.udf", fake_udf)
@patch("src.inference_streaming.ModelWrapper")
@patch("src.inference_streaming.SparkSession")
def test_streaming_setup(mock_spark, mock_model):

    df = MagicMock()
    df.select.return_value = df
    df.withColumn.return_value = df

    df.writeStream.format.return_value.option.return_value.start.return_value.awaitTermination.return_value = None

    mock_spark.builder.getOrCreate.return_value.readStream.format.return_value.option.return_value.load.return_value = df

    run_stream()

    assert True
