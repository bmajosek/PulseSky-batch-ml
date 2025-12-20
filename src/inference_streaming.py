"""Streaming inference pipeline for real-time sentiment analysis."""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_timestamp, window,
    when, count, avg, sum as spark_sum
)
from pyspark.sql.types import StructType, StructField, StringType

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    MODEL_PATH,
    KAFKA_BROKERS,
    KAFKA_TOPIC,
    CHECKPOINT_PATH,
    S3_GOLD_1M_PATH,
)

kafka_schema = StructType([
    StructField("language", StringType()),
    StructField("text", StringType()),
    StructField("did", StringType()),
    StructField("timestamp", StringType()),
])


def run_streaming_inference():

    spark = (
        SparkSession.builder
        .appName("sentiment-streaming-1m")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    kafka_df = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_BROKERS)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "earliest")
        .option("failOnDataLoss", "false")
        .load()
    )

    parsed_df = (
        kafka_df
        .selectExpr("CAST(value AS STRING) AS json_str")
        .withColumn("json", from_json(col("json_str"), kafka_schema))
        .select(
            col("json.text").alias("text"),
            col("json.language").alias("language"),
            col("json.timestamp").alias("timestamp"),
        )
        .filter(col("text").isNotNull())
        .filter(col("text") != "")
    )

    def write_batch(batch_df, batch_id):
        print(f"\n🔥 BATCH {batch_id}")

        if batch_df.isEmpty():
            print("Empty batch")
            return

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()

        def infer_partition(rows):
            for r in rows:
                inputs = tokenizer(
                    r.text,
                    truncation=True,
                    padding="max_length",
                    max_length=128,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    logits = model(**inputs).logits
                    label_id = logits.argmax(dim=1).item()
                    sentiment = model.config.id2label[label_id]

                yield (r.timestamp, sentiment)

        scored_df = (
            batch_df
            .rdd
            .mapPartitions(infer_partition)
            .toDF(["timestamp", "sentiment"])
            .withColumn("event_time", to_timestamp("timestamp"))
            .withColumn(
                "sentiment_score",
                when(col("sentiment") == "negative", -1)
                .when(col("sentiment") == "neutral", 0)
                .when(col("sentiment") == "positive", 1)
            )
        )

        gold_agg = (
            scored_df
            .withWatermark("event_time", "10 minutes")
            .groupBy(window(col("event_time"), "1 minute"))
            .agg(
                count("*").alias("post_count"),
                avg("sentiment_score").alias("avg_sentiment"),
                spark_sum(when(col("sentiment") == "positive", 1).otherwise(0)).alias("positive_count"),
                spark_sum(when(col("sentiment") == "neutral", 1).otherwise(0)).alias("neutral_count"),
                spark_sum(when(col("sentiment") == "negative", 1).otherwise(0)).alias("negative_count"),
            )
        )

        gold_agg.write.mode("append").parquet(S3_GOLD_1M_PATH)
        print("✅ Written GOLD 1m")

    (
        parsed_df
        .writeStream
        .foreachBatch(write_batch)
        .option("checkpointLocation", CHECKPOINT_PATH + "_1m")
        .trigger(processingTime="10 seconds")
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    run_streaming_inference()
