import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
import warnings
import mlflow
import dagshub

warnings.filterwarnings('ignore')

X, y = make_classification(n_samples=1000, n_features=10, n_informative=2, n_redundant=8,
                           weights=[0.9, 0.1], flip_y=0, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

lr_params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

lr = LogisticRegression(**lr_params)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
report_lr = classification_report(y_test, y_pred_lr, output_dict=True)

xgb_params = {
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 3,
    "random_state": 8888,
}

xgb = XGBClassifier(**xgb_params)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
report_xgb = classification_report(y_test, y_pred_xgb, output_dict=True)


dagshub.init(repo_owner='rohansb10', repo_name='my-first-repo', mlflow=True)

mlflow.set_experiment("rohanbarad")
mlflow.set_tracking_uri(uri="https://dagshub.com/rohansb10/my-first-repo.mlflow/")

with mlflow.start_run(run_name="testing model"):
    mlflow.log_params(lr_params)
    mlflow.log_metrics({
        'accuracy_lr': report_lr['accuracy'],
        'recall_class_0_lr': report_lr['0']['recall'],
        'recall_class_1_lr': report_lr['1']['recall'],
        'f1_score_macro_lr': report_lr['macro avg']['f1-score']
    })
    mlflow.sklearn.log_model(lr, "Logistic Regression")

    mlflow.log_params(xgb_params)
    mlflow.log_metrics({
        'accuracy_xgb': report_xgb['accuracy'],
        'recall_class_0_xgb': report_xgb['0']['recall'],
        'recall_class_1_xgb': report_xgb['1']['recall'],
        'f1_score_macro_xgb': report_xgb['macro avg']['f1-score']
    })
    mlflow.xgboost.log_model(xgb, "XGBoost")
