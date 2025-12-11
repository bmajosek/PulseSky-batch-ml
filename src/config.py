# config.py

class Config:
    MODEL_PATH = "./models/sentiment_model"

    # S3 paths
    SILVER_PATH = "s3://bigdata/silver/posts/"
    GOLD_PATH = "s3://bigdata/gold/sentiment/"

    # Kafka streaming
    KAFKA_BROKERS = "localhost:9092"
    KAFKA_TOPIC = "posts_clean"

cfg = Config()
