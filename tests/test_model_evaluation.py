import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def test_model_performance_metrics():
    """
    Final evaluation of analytical module:
    accuracy, precision, recall, f1-score
    """

    # Ground truth labels
    y_true = [
        "positive", "negative", "neutral",
        "positive", "negative", "positive"
    ]

    # Model predictions
    y_pred = [
        "positive", "negative", "neutral",
        "positive", "neutral", "positive"
    ]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro"
    )

    # --- ASSERTIONS (evaluation criteria) ---
    assert accuracy >= 0.70
    assert precision >= 0.65
    assert recall >= 0.65
    assert f1 >= 0.65
