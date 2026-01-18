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
        print(OPENSEARCH_HOST)
        self.client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_auth=(OPENSEARCH_USER, OPENSEARCH_PASSWORD),
            use_ssl=OPENSEARCH_USE_SSL,
            verify_certs=True,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )

        self.index_name = OPENSEARCH_SENTIMENT_INDEX
        self._ensure_index_exists()

    def _ensure_index_exists(self):
        if self.client.indices.exists(index=self.index_name):
            return

        self.client.indices.create(
            index=self.index_name,
            body={
                "settings": {
                    "number_of_shards": 3,
                    "number_of_replicas": 2,
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
            },
        )

    def write_predictions(self, predictions, batch_id):
        actions = [
            {
                "_index": self.index_name,
                "_source": {
                    "timestamp": datetime.utcnow().isoformat(),
                    "event_time": p.get("event_time"),
                    "text": p.get("text"),
                    "language": p.get("language"),
                    "sentiment": p.get("sentiment"),
                    "sentiment_score": float(p.get("sentiment_score", 0)),
                    "batch_id": str(batch_id),
                },
            }
            for p in predictions
        ]

        helpers.bulk(
            self.client,
            actions,
            chunk_size=OPENSEARCH_BATCH_SIZE,
            raise_on_error=False,
        )

    def write_aggregated_metrics(self, metrics, batch_id):
        self.client.index(
            index=f"{self.index_name}-metrics",
            body={
                "timestamp": datetime.utcnow().isoformat(),
                "batch_id": str(batch_id),
                **metrics,
            },
        )

    def close(self):
        self.client.close()
