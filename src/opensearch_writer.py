from datetime import datetime
from opensearchpy import OpenSearch, helpers
from src.config import (
    OPENSEARCH_HOST,
    OPENSEARCH_PORT,
    OPENSEARCH_USER,
    OPENSEARCH_PASSWORD,
    OPENSEARCH_SENTIMENT_INDEX,
    OPENSEARCH_BATCH_SIZE,
    OPENSEARCH_USE_SSL,
)


class OpenSearchWriter:

    def __init__(self):
        self.client = OpenSearch(
            hosts=[
                {
                    "host": OPENSEARCH_HOST,
                    "port": OPENSEARCH_PORT,
                }
            ],
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
            use_ssl=OPENSEARCH_USE_SSL,
            verify_certs=False,
            ssl_show_warn=False,
        )
        self.index_name = OPENSEARCH_SENTIMENT_INDEX
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        if not self.client.indices.exists(index=self.index_name):
            index_body = {
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 1,
                    "index.refresh_interval": "5s",
                },
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "event_time": {"type": "date"},
                        "text": {"type": "text"},
                        "language": {"type": "keyword"},
                        "sentiment": {"type": "keyword"},
                        "sentiment_score": {"type": "float"},
                        "batch_id": {"type": "keyword"},
                    }
                },
            }
            self.client.indices.create(
                index=self.index_name, body=index_body
            )

    def write_predictions(self, predictions, batch_id):
        if not predictions:
            return 0

        actions = []
        for pred in predictions:
            doc = {
                "_index": self.index_name,
                "_source": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_time": pred.get("event_time"),
                    "text": pred.get("text"),
                    "language": pred.get("language"),
                    "sentiment": pred.get("sentiment"),
                    "sentiment_score": float(pred.get("sentiment_score", 0)),
                    "batch_id": batch_id,
                },
            }
            actions.append(doc)

        try:
            success, failed = helpers.bulk(
                self.client,
                actions,
                chunk_size=OPENSEARCH_BATCH_SIZE,
                raise_on_error=False,
            )
            return success
        except Exception as e:
            print(f"Error: {e}")
            return 0

    def write_aggregated_metrics(self, metrics, batch_id):
        doc = {
            "_index": f"{self.index_name}-metrics",
            "_source": {
                "timestamp": datetime.utcnow().isoformat(),
                "batch_id": batch_id,
                "window": metrics.get("window"),
                "post_count": int(metrics.get("post_count", 0)),
                "avg_sentiment": float(metrics.get("avg_sentiment", 0)),
                "positive_count": int(metrics.get("positive_count", 0)),
                "neutral_count": int(metrics.get("neutral_count", 0)),
                "negative_count": int(metrics.get("negative_count", 0)),
            },
        }

        try:
            self.client.index(
                index=f"{self.index_name}-metrics",
                body=doc["_source"],
            )
        except Exception as e:
            print(f"Error: {e}")

    def close(self):
        if self.client:
            self.client.close()
