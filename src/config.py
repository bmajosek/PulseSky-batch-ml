MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL_OUTPUT_DIR = "models/sentiment_roberta"
MODEL_PATH = "models/sentiment_roberta/checkpoint-148"

S3_BRONZE_PATH = "s3a://bigdata-bluesky-sentiment/bluesky_raw/"
S3_SILVER_PATH = "s3a://bigdata-bluesky-sentiment/silver/posts/"
S3_SILVER_SENTIMENT_PATH = "s3a://bigdata-bluesky-sentiment/silver/annotations/sentiment/"
S3_GOLD_PATH = "s3a://bigdata-bluesky-sentiment/gold/sentiment_batch/"
S3_GOLD_1M_PATH = "s3a://bigdata-bluesky-sentiment/gold/sentiment_1m/"

SUPPORTED_LANGS = ["en"]

KAFKA_BROKERS = "54.226.214.16:9092"
KAFKA_TOPIC = "blusky"
CHECKPOINT_PATH = "s3a://bigdata-bluesky-sentiment/checkpoints/sentiment_stream_v2/"

# OpenSearch configuration
OPENSEARCH_HOST = "search-dashboard-search-XXXX.aos.us-east-1.on.aws"
OPENSEARCH_PORT = 443
OPENSEARCH_USE_SSL = True
OPENSEARCH_USER = "XXXX"
OPENSEARCH_PASSWORD = "XXXX"
OPENSEARCH_SENTIMENT_INDEX = "sentiment-predictions-gold"
OPENSEARCH_BATCH_SIZE = 500


