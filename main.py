import numpy as np
import pandas as pd
from src.data_loader import load_data
from src.eda import perform_eda
from src.feature_engineering import engineer_features
from src.model import optimize_model
from src.visualization import plot_feature_importance, plot_shap_values

# To ignore optuna FutureWarnings
import warnings
warnings.filterwarnings("ignore")

def main():
    # Set random seed for reproducibility
    np.random.seed(42)

    # Load data
    data, _ = load_data()

    # Perform EDA
    perform_eda(data)

    # Engineer features
    data = engineer_features(data)

    # Print data info after feature engineering
    print("\nData info after feature engineering:")
    print(data.info())

    # Prepare data for modeling
    X = data.drop(['CHURN', 'SUBSCRIBER_ID'], axis=1)
    y = data['CHURN']

    # Print X info
    print("\nFeature set (X) info:")
    print(X.info())

    # Train and optimize model
    final_model, X_test = optimize_model(X, y)

    # Analyze feature importance
    plot_feature_importance(final_model, X_test)
    plot_shap_values(final_model, X_test)


if __name__ == "__main__":
    main()