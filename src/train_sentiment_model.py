"""Training script for sentiment analysis model."""
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from src.config import MODEL_NAME, MODEL_OUTPUT_DIR
from src.utils import prepare_pandas_dataset

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def train_model(pdf):
    pdf = prepare_pandas_dataset(pdf)
    pdf["label"] = pdf["sentiment_label"].map(LABEL2ID)
    pdf = pdf.dropna(subset=["label"])

    train_df, eval_df = train_test_split(
        pdf,
        test_size=0.2,
        stratify=pdf["label"],
        random_state=42,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(
            batch["text_clean"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )

    train_ds = Dataset.from_pandas(train_df[["text_clean", "label"]])
    eval_ds = Dataset.from_pandas(eval_df[["text_clean", "label"]])

    train_ds = train_ds.map(tokenize, batched=True).remove_columns(["text_clean"])
    eval_ds = eval_ds.map(tokenize, batched=True).remove_columns(["text_clean"])

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=16,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(MODEL_OUTPUT_DIR)


if __name__ == "__main__":
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.getOrCreate()

    df = spark.read.parquet(
        "s3a://bigdata-bluesky-sentiment/silver/annotations/sentiment/"
    )

    pdf = df.toPandas()

    train_model(pdf)
