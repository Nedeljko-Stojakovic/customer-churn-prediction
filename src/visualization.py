import matplotlib.pyplot as plt
import numpy as np
import shap

def plot_feature_importance(model, X_test):
    feature_importance = model.feature_importances_
    sorted_idx = np.argsort(feature_importance)
    pos = np.arange(sorted_idx.shape[0]) + .5

    plt.figure(figsize=(12, 6))
    plt.barh(pos, feature_importance[sorted_idx], align='center')
    plt.yticks(pos, X_test.columns[sorted_idx])
    plt.title('Feature Importance (MDI)')
    plt.savefig('outputs/feature_importance.png')
    plt.close()

def plot_shap_values(model, X_test):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    plt.title('SHAP Feature Importance')
    plt.tight_layout()
    plt.savefig('outputs/shap_importance.png')
    plt.close()