import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score


def generate_synthetic_data(n=500):
    np.random.seed(42)
    return pd.DataFrame({
        "cvss_v3": np.random.uniform(4.0, 10.0, size=n),
        "exploit_available": np.random.choice([0, 1], size=n),
        "asset_exposure": np.random.uniform(0.1, 1.0, size=n),
        "historical_signal": np.random.randint(0, 2, size=n)
    })


def train_models(df):
    X = df[["cvss_v3", "asset_exposure", "historical_signal"]]
    y = df["exploit_available"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Logistic Regression
    lr = LogisticRegression(max_iter=200)
    lr.fit(X_train, y_train)
    preds_lr = lr.predict(X_test)
    probs_lr = lr.predict_proba(X_test)[:, 1]

    print("Logistic Regression Results:")
    print(classification_report(y_test, preds_lr))
    print("LR ROC-AUC:", roc_auc_score(y_test, probs_lr))

    # XGBoost
    xgb = XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    preds_xgb = xgb.predict(X_test)
    probs_xgb = xgb.predict_proba(X_test)[:, 1]

    print("\nXGBoost Results:")
    print(classification_report(y_test, preds_xgb))
    print("XGB ROC-AUC:", roc_auc_score(y_test, probs_xgb))

    return lr, xgb, X_test, y_test, probs_xgb
