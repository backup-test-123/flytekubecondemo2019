import pandas as pd
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve


def calculate_roc_curve(ground_truths, predictions, pos_label_idx):
    if (
        not ground_truths
        or not predictions
        or pos_label_idx not in range(len(predictions[0]))
    ):
        return [], [], []

    predictions = [p[pos_label_idx] for p in predictions]
    fpr, tpr, roc_thresholds = roc_curve(
        y_true=ground_truths, y_score=predictions, pos_label=pos_label_idx
    )
    return fpr, tpr, roc_thresholds


def calculate_precision_recall_curve(ground_truths, predictions, pos_label_idx):
    if (
        not ground_truths
        or not predictions
        or pos_label_idx not in range(len(predictions[0]))
    ):
        return [], [], []

    predictions = [p[pos_label_idx] for p in predictions]
    precisions, recalls, prc_thresholds = precision_recall_curve(
        y_true=ground_truths, probas_pred=predictions, pos_label=pos_label_idx
    )
    return precisions, recalls, prc_thresholds


def calculate_accuracy(ground_truths, predictions, threshold):
    opt_pred_labels = [1 if (pred >= threshold) else 0 for pred in predictions]
    accuracy = sum(1 for x, y in zip(ground_truths, opt_pred_labels) if x == y) / float(
        len(ground_truths)
    )
    return accuracy


def calculate_cutoff_youdens_j(tpr, fpr, thresholds):
    """
    This function calculates the best threshold considering the ROC curve
    represented by (tpr, fpr, thresholds)
    """
    # Reference: https://stackoverflow.com/questions/28719067/roc-curve-and-cut-off-point-python
    j_scores = tpr - fpr
    j_ordered = sorted(zip(j_scores, thresholds))
    return j_ordered[-1][1]


def export_results(path, columns, index_col):
    df = pd.DataFrame({k: pd.Series(v) for k, v in columns.items()})

    if index_col is not None and index_col in columns.keys():
        df.set_index(index_col)
    else:
        print(f"Specified index column {index_col} does not exist!")
    df.to_csv(path, na_rep="NA")
