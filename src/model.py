import optuna
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score


def optimize_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    def objective(trial):
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.1),
            'n_estimators': 1000,
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_uniform('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.5, 1.0),
            'early_stopping_rounds': 50
        }

        model = xgb.XGBClassifier(**param, eval_metric='logloss')
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

        # Get the best score
        best_score = min(model.evals_result()['validation_0']['logloss'])
        return best_score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    print('Best parameters:', study.best_params)
    print('Best validation score:', study.best_value)

    # Train final model
    best_params = study.best_params
    best_params['n_estimators'] = 1000
    best_params['early_stopping_rounds'] = 50
    final_model = xgb.XGBClassifier(**best_params, use_label_encoder=False, eval_metric='logloss')

    # Use the fit method with eval_set for early stopping
    final_model.fit(X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False)

    # Evaluate on test set
    y_pred = final_model.predict(X_test)
    y_pred_proba = final_model.predict_proba(X_test)[:, 1]

    print("Test AUC:", roc_auc_score(y_test, y_pred_proba))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    return final_model, X_test