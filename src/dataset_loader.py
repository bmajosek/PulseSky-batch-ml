"""Data loader for reading posts from S3 Bronze layer."""
from pyspark.sql import DataFrame
from pyspark.sql.functions import col, element_at

from src.config import S3_BRONZE_PATH, SUPPORTED_LANGS


class DatasetLoader:
    def __init__(self, spark):
        self.spark = spark

    def load_posts(self):
        df = self.spark.read.parquet(S3_BRONZE_PATH)

        df = df.select(
            col("did").alias("post_id"),
            col("commit.record.createdAt").alias("created_at"),
            col("commit.record.text").alias("text"),

            element_at(col("commit.record.langs"), 1).alias("language"),
        )

        df = df.filter(col("text").isNotNull())

        if SUPPORTED_LANGS:
            df = df.filter(col("language").isin(SUPPORTED_LANGS))

        return df
