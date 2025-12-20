"""Batch inference pipeline for sentiment annotation."""
import pandas as pd
from pyspark.sql import SparkSession
from transformers import pipeline
import torch

from src.config import MODEL_PATH, S3_SILVER_SENTIMENT_PATH
from src.dataset_loader import DatasetLoader
from src.utils import prepare_pandas_dataset


def create_spark_session():
    return (
        SparkSession.builder
        .appName("sentiment-weak-annotation")
        .master("local[*]")
        .config(
            "spark.jars.packages",
            "org.apache.hadoop:hadoop-aws:3.3.4,"
            "com.amazonaws:aws-java-sdk-bundle:1.12.662"
        )
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config(
            "spark.hadoop.fs.s3a.aws.credentials.provider",
            "com.amazonaws.auth.DefaultAWSCredentialsProviderChain"
        )
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .getOrCreate()
    )


def run_inference():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    try:
        loader = DatasetLoader(spark)
        df = loader.load_posts()

        if df.limit(1).count() == 0:
            print("No posts found — nothing to annotate.")
            return

        pdf = df.toPandas()

        pdf["created_at"] = pd.to_datetime(
            pdf["created_at"],
            utc=True,
            errors="coerce"
        )

        pdf = prepare_pandas_dataset(pdf)

        if pdf.empty:
            print("Dataset empty after preprocessing — exiting.")
            return

        device = 0 if torch.cuda.is_available() else -1

        classifier = pipeline(
            "sentiment-analysis",
            model=MODEL_PATH,
            device=device,
        )

        results = classifier(
            pdf["text_clean"].tolist(),
            batch_size=32,
            truncation=True,
        )

        pdf["sentiment_label"] = [r["label"] for r in results]
        pdf["sentiment_score"] = [r["score"] for r in results]

        silver_df = spark.createDataFrame(
            pdf[
                [
                    "post_id",
                    "created_at",
                    "language",
                    "text_clean",
                    "sentiment_label",
                    "sentiment_score",
                ]
            ]
        )

        silver_df.write.mode("overwrite").parquet(
            S3_SILVER_SENTIMENT_PATH
        )

        print("\n=== SILVER DATA SAMPLE ===")
        silver_df.show(20, truncate=False)
        print(f"Total annotated rows: {silver_df.count()}")

    finally:
        spark.stop()


if __name__ == "__main__":
    run_inference()
