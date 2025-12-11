# streaming_inference.py
from pyspark.sql import SparkSession, functions as F, types as T
from src.model_wrapper import ModelWrapper
from src.config import cfg

def run_stream():
    spark = SparkSession.builder.appName("StreamInference").getOrCreate()

    schema = T.StructType([
        T.StructField("text", T.StringType()),
        T.StructField("platform", T.StringType())
    ])

    df = spark.readStream.format("kafka") \
        .option("kafka.bootstrap.servers", cfg.KAFKA_BROKERS) \
        .option("subscribe", cfg.KAFKA_TOPIC) \
        .load()

    parsed = df.select(F.from_json(F.col("value").cast("string"), schema).alias("d")).select("d.*")

    model = ModelWrapper(cfg.MODEL_PATH)
    pred = F.udf(lambda t: model.predict(t)["sentiment"], T.StringType())

    out = parsed.withColumn("sentiment", pred("text"))

    q = out.writeStream \
        .format("parquet") \
        .option("path", cfg.GOLD_PATH) \
        .option("checkpointLocation", "./chk_stream") \
        .start()

    q.awaitTermination()

if __name__ == "__main__":
    run_stream()
