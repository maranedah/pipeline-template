from sklearn.metrics import accuracy_score


def get_metrics(X_train, y_train, X_test, y_test, model, baseline_model):
    sets_to_compare = {
        "train": (model.predict(X_train), y_train),
        "test": (model.predict(X_test), y_test),
        # "baseline": (baseline_model.predict(X_test), y_test),
    }
    metrics = {}
    for set_name, (y_hat, y_true) in sets_to_compare.items():
        metrics.update({f"{set_name}_accuracy": accuracy_score(y_true, y_hat)})

    return metrics
