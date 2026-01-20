from src.opensearch_writer import OpenSearchWriter


w = OpenSearchWriter()
w.write_predictions(
    [{
        "event_time": "2025-01-01T00:00:00Z",
        "text": "to dziala",
        "language": "pl",
        "sentiment": "positive",
        "sentiment_score": 1.0,
    }],
    batch_id=1,
)
print("OK")
