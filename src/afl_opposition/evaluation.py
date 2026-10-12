"""Threshold-specific confusion matrices alongside probability-based evaluation."""

from math import isfinite

import pandas as pd
from sklearn.metrics import confusion_matrix


def _ratio(numerator, denominator) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def confusion_report(predictions: pd.DataFrame, *, threshold=0.5, probability_columns=None) -> pd.DataFrame:
    """Return raw and possession-weighted binary counts and derived metrics.

    Positive means a subsequent recorded shot. Predictions are positive when
    score >= threshold, including ties. Matrix rows are actual classes and
    columns are predictions: [[TN, FP], [FN, TP]]. Undefined rates remain NaN.
    This function only evaluates existing scores; it never fits a model or
    chooses a threshold. Custom probability_columns maps model names to columns.
    """
    if not isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The decision threshold must be finite and between 0 and 1.")
    if probability_columns is None:
        probability_columns = {"constant": "constant_probability", "logistic": "shot_probability"}
    if not probability_columns:
        raise ValueError("At least one model probability column is required.")
    required = {"split", "target", "sample_weight", *probability_columns.values()}
    if required - set(predictions) or predictions.empty:
        raise ValueError("A nonempty prediction table with labels, weights and probabilities is required.")
    if predictions[list(required)].isna().any().any() or not predictions.target.isin([0, 1]).all():
        raise ValueError("Labels must be binary and evaluation fields must not be missing.")
    if (predictions.sample_weight <= 0).any() or not isfinite(float(predictions.sample_weight.sum())):
        raise ValueError("Sample weights must be finite and strictly positive.")
    for column in probability_columns.values():
        if not predictions[column].between(0, 1).all():
            raise ValueError("Model probabilities must be finite and between 0 and 1.")
    rows = []
    for split, group in predictions.groupby("split", sort=True):
        for model, column in probability_columns.items():
            predicted = group[column].ge(threshold)
            for weighting, weights in [("unweighted", None), ("possession_weighted", group.sample_weight)]:
                tn, fp, fn, tp = confusion_matrix(
                    group.target, predicted, labels=[0, 1], sample_weight=weights,
                ).ravel()
                total = tn + fp + fn + tp
                recall, specificity = _ratio(tp, tp + fn), _ratio(tn, tn + fp)
                rows.append(dict(
                    split=split, model=model, weighting=weighting, threshold=float(threshold),
                    tn=tn, fp=fp, fn=fn, tp=tp, total=total,
                    accuracy=_ratio(tp + tn, total), precision=_ratio(tp, tp + fp),
                    recall=recall, specificity=specificity,
                    f1=_ratio(2 * tp, 2 * tp + fp + fn),
                    balanced_accuracy=(recall + specificity) / 2,
                    predicted_shot_rate=_ratio(tp + fp, total),
                ))
    return pd.DataFrame(rows)
