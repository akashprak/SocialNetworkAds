import pickle
from pathlib import Path
from sklearn.metrics import accuracy_score, roc_auc_score

def save_object(file_path, obj):
    """
    for saving objects to the given path
    """
    dir_path = Path(file_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)

    with open(file_path, "wb") as file_obj:
        pickle.dump(obj, file_obj)

def evaluate_model(model, X_test, y_test):
    "Returns: acc_score, roc_auc_score"
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return acc, roc_auc