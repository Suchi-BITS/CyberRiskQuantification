from sklearn.isotonic import IsotonicRegression
import shap
import matplotlib.pyplot as plt


def calibrate_model(probs, labels):
    iso_reg = IsotonicRegression(out_of_bounds="clip")
    iso_reg.fit(probs, labels)
    calibrated = iso_reg.transform(probs)
    return calibrated


def explain_model(model, X_test):
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test)
