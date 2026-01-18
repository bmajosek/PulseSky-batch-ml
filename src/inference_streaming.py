import os
import torch
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, from_json, to_timestamp, when,
    window, count, avg, sum as spark_sum
)
from pyspark.sql.types import StructType, StructField, StringType
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import (
    MODEL_PATH,
    KAFKA_BROKERS,
    KAFKA_TOPIC,
    CHECKPOINT_PATH,
    S3_GOLD_1M_PATH,
)
from src.opensearch_writer import OpenSearchWriter

kafka_schema = StructType([
    StructField("language", StringType()),
    StructField("text", StringType()),
    StructField("timestamp", StringType()),
])

def run_streaming_inference():

    spark = (
        SparkSession.builder
        .appName("sentiment-streaming")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = os.path.abspath(MODEL_PATH)
    print(f" Loading tokenizer from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

    print(f" Loading model from: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True
    ).to(device)
    model.eval()

    os_writer = OpenSearchWriter()

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
        print(f"\n BATCH {batch_id}")

        if batch_df.isEmpty():
            print("Empty batch")
            return

        rows = batch_df.collect()
        predictions = []

        for r in rows:
            inputs = tokenizer(
                r.text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                label_id = logits.argmax(dim=1).item()
                sentiment = model.config.id2label[label_id]

            score = {"negative": -1, "neutral": 0, "positive": 1}[sentiment]

            predictions.append({
                "timestamp": r.timestamp,
                "text": r.text,
                "language": r.language,
                "sentiment": sentiment,
                "sentiment_score": score,
            })

        if predictions:
            os_writer.write_predictions(predictions, batch_id)
            print(f"Written {len(predictions)} docs to OpenSearch")

        if predictions:
            scored_df = spark.createDataFrame(predictions)

            scored_df = (
                scored_df
                .withColumn("event_time", to_timestamp(col("timestamp")))
                # jeśli timestamp bywa nieparsowalny -> event_time będzie null
                .filter(col("event_time").isNotNull())
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
            print(f"Written GOLD 1m to S3: {S3_GOLD_1M_PATH}")

    (
        parsed_df
        .writeStream
        .foreachBatch(write_batch)
        .option("checkpointLocation", CHECKPOINT_PATH)
        .trigger(processingTime="10 seconds")
        .start()
        .awaitTermination()
    )

if __name__ == "__main__":
    run_streaming_inference()
