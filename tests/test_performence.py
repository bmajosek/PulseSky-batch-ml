import pandas as pd
import time
from src.utils import prepare_pandas_dataset


def test_pipeline_performance_large_batch():
    """
    Performance test: preprocessing 10k records under time limit
    """

    df = pd.DataFrame({
        "text": ["This is a performance test sentence"] * 10_000
    })

    start = time.time()
    result = prepare_pandas_dataset(df)
    duration = time.time() - start

    assert len(result) == 10_000
    assert duration < 5.0
