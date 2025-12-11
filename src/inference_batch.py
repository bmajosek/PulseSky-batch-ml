# batch_inference.py
from pyspark.sql import SparkSession, functions as F, types as T
from src.model_wrapper import ModelWrapper
from src.config import cfg

def run_batch():
    spark = SparkSession.builder.appName("BatchInference").getOrCreate()
    df = spark.read.parquet(cfg.SILVER_PATH)

    model = ModelWrapper(cfg.MODEL_PATH)

    pred = F.udf(lambda t: model.predict(t)["sentiment"], T.StringType())

    scored = df.withColumn("sentiment", pred("text"))
    scored.write.mode("overwrite").parquet(cfg.GOLD_PATH)

if __name__ == "__main__":
    run_batch()
