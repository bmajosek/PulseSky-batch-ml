"""Model evaluation script."""
from sklearn.metrics import classification_report
from transformers import pipeline
from pyspark.sql import SparkSession

from src.config import MODEL_PATH
from src.utils import prepare_pandas_dataset


def create_spark_session() -> SparkSession:
    return (
        SparkSession.builder
        .appName("sentiment-evaluation")
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
        .getOrCreate()
    )


def evaluate_model(pdf):
    pdf = prepare_pandas_dataset(pdf)
    clf = pipeline(
        "sentiment-analysis",
        model=MODEL_PATH,
    )

    preds = clf(
        pdf["text_clean"].tolist(),
        batch_size=32,
        truncation=True,
    )

    pdf["prediction"] = [p["label"] for p in preds]

    print(
        classification_report(
            pdf["sentiment_label"],
            pdf["prediction"],
            digits=3,
        )
    )


if __name__ == "__main__":

    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")

    df = spark.read.parquet(
        "s3a://bigdata-bluesky-sentiment/silver/annotations/sentiment/"
    )

    pdf = df.sample(fraction=0.2, seed=42).toPandas()

    evaluate_model(pdf)

    spark.stop()
