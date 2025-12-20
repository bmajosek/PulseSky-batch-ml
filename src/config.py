"""Configuration for sentiment analysis pipeline."""

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL_OUTPUT_DIR = "models/sentiment_roberta"
MODEL_PATH = "models/sentiment_roberta/checkpoint-174"

S3_BRONZE_PATH = "s3a://bigdata-bluesky-sentiment/bluesky_raw/"
S3_SILVER_PATH = "s3a://bigdata-bluesky-sentiment/silver/posts/"
S3_SILVER_SENTIMENT_PATH = "s3a://bigdata-bluesky-sentiment/silver/annotations/sentiment/"
S3_GOLD_PATH = "s3a://bigdata-bluesky-sentiment/gold/sentiment_batch/"
S3_GOLD_1M_PATH = "s3a://bigdata-bluesky-sentiment/gold/sentiment_1m/"

SUPPORTED_LANGS = ["en"]

KAFKA_BROKERS = "54.226.214.16:9092"
KAFKA_TOPIC = "bluesky"
CHECKPOINT_PATH = "s3a://bigdata-bluesky-sentiment/checkpoints/sentiment_stream_dbg/"

