"""Utility functions for data preprocessing and text cleaning."""
import re

import pandas as pd


URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
MENTION_PATTERN = re.compile(r"@\w+")
WHITESPACE_PATTERN = re.compile(r"\s+")


def clean_text(text: str):
    if not text:
        return ""

    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = MENTION_PATTERN.sub("", text)
    text = WHITESPACE_PATTERN.sub(" ", text)

    return text.strip()


def prepare_pandas_dataset(pdf):
    pdf = pdf.copy()

    if "text" in pdf.columns:
        pdf["text_clean"] = pdf["text"].apply(clean_text)
    elif "text_clean" in pdf.columns:
        pdf["text_clean"] = pdf["text_clean"].astype(str).apply(clean_text)
    else:
        raise KeyError(
            "prepare_pandas_dataset expects a 'text' or 'text_clean' column"
        )

    pdf = pdf[pdf["text_clean"].str.len() > 3]

    return pdf
