# dataset_loader.py
import pandas as pd
from pyspark.sql import SparkSession
from src.config import cfg

class DatasetLoader:
    def __init__(self):
        self.spark = SparkSession.builder.appName("Loader").getOrCreate()

    def load_silver(self, limit=5000):
        df = self.spark.read.parquet(cfg.SILVER_PATH)
        pdf = df.select("text", "sentiment_label").dropna().toPandas()
        pdf = pdf.rename(columns={"sentiment_label": "label"})
        return pdf.sample(n=min(limit, len(pdf)), random_state=42)
