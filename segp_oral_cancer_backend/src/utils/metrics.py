from sklearn.metrics import f1_score, roc_auc_score

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def binary_auc(y_true, y_prob):
    try:
        return roc_auc_score(y_true, y_prob)
    except Exception:
        return float("nan")
