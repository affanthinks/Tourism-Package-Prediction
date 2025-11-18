# for data manipulation
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# for model training, tuning, and evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, recall_score
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)
# for model serialization
import joblib
# for creating a folder
import os
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("mlops-training-experiment")

api = HfApi()


Xtrain_path = "hf://datasets/affanthinks/Tourism-Package-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/affanthinks/Tourism-Package-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/affanthinks/Tourism-Package-Prediction/ytrain.csv"
ytest_path = "hf://datasets/affanthinks/Tourism-Package-Prediction/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path)
ytest = pd.read_csv(ytest_path)


# List of numerical features in the dataset
numeric_features = [
    'Age',              #
    'CityTier',               #
    'DurationOfPitch',            #
    'NumberOfPersonVisiting',           #
    'NumberOfFollowups',     #
    'PreferredPropertyStar',         #
    'NumberOfTrips',    #
    'Passport', #
    'PitchSatisfactionScore',  #
    'OwnCar',  #
    'NumberOfChildrenVisiting',  #
    'MonthlyIncome'   #
]

# List of categorical features in the dataset
categorical_features = [
    'TypeofContact', #
    'Occupation',#
    'Gender', #
    'ProductPitched', #
    'MaritalStatus', #
    'Designation'#
]


# Set the clas weight to handle class imbalance
class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]
class_weight

# Define the preprocessing steps
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# Define base XGBoost model
xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

# --- Randomized search configuration ---
N_ITER = 50  # number of random samples to draw; increase if you have the time/compute
SCORER = "roc_auc"  # good default for imbalanced binary classification; change if needed

# Convert your original discrete grid to distributions
param_dist = {
    'xgbclassifier__n_estimators': randint(50, 151),          # sample from 50..150 inclusive
    'xgbclassifier__max_depth': randint(2, 5),                # 2..4
    'xgbclassifier__colsample_bytree': uniform(0.4, 0.2),    # 0.4 .. 0.6
    'xgbclassifier__colsample_bylevel': uniform(0.4, 0.2),   # 0.4 .. 0.6
    'xgbclassifier__learning_rate': uniform(0.01, 0.09),     # 0.01 .. 0.10
    'xgbclassifier__reg_lambda': uniform(0.4, 0.2),          # 0.4 .. 0.6
}

# Build pipeline (same as you had)
model_pipeline = make_pipeline(preprocessor, xgb_model)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=model_pipeline,
    param_distributions=param_dist,
    n_iter=N_ITER,
    scoring=SCORER,
    cv=5,
    random_state=42,
    n_jobs=-1,
    verbose=1,
    return_train_score=False
)

# Run with MLflow logging similar to your original nested-run approach
with mlflow.start_run():
    mlflow.log_param("search_type", "RandomizedSearchCV")
    mlflow.log_param("n_iter", N_ITER)
    mlflow.log_param("scorer", SCORER)

    # Fit the randomized search
    random_search.fit(Xtrain, ytrain)

    # Log all trial results (one nested run per sampled param set)
    results = random_search.cv_results_
    for i in range(len(results['params'])):
        trial_params = results['params'][i]
        mean_score = results['mean_test_score'][i]
        std_score = results['std_test_score'][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(trial_params)
            mlflow.log_metric("mean_test_score", float(mean_score))
            mlflow.log_metric("std_test_score", float(std_score))

    # Log best params to main run
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_score", float(best_score))

    # Get best estimator and evaluate with your threshold
    best_model = random_search.best_estimator_
    classification_threshold = 0.45

    # Predict probabilities and threshold
    y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
    y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

    y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
    y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

    # Ensure ytrain/ytest are 1-D arrays (if they are DataFrames)
    ytrain_vec = np.ravel(ytrain)
    ytest_vec = np.ravel(ytest)

    train_report = classification_report(ytrain_vec, y_pred_train, output_dict=True, zero_division=0)
    test_report = classification_report(ytest_vec, y_pred_test, output_dict=True, zero_division=0)

    # Defensive logging: if class '1' missing, fallback to 0s
    def safe_get_report_metric(report, cls, metric):
        try:
            return float(report[str(cls)][metric])
        except Exception:
            return 0.0

    mlflow.log_metrics({
        "train_accuracy": float(train_report.get('accuracy', 0.0)),
        "train_precision_class1": safe_get_report_metric(train_report, 1, 'precision'),
        "train_recall_class1": safe_get_report_metric(train_report, 1, 'recall'),
        "train_f1_class1": safe_get_report_metric(train_report, 1, 'f1-score'),
        "test_accuracy": float(test_report.get('accuracy', 0.0)),
        "test_precision_class1": safe_get_report_metric(test_report, 1, 'precision'),
        "test_recall_class1": safe_get_report_metric(test_report, 1, 'recall'),
        "test_f1_class1": safe_get_report_metric(test_report, 1, 'f1-score')
    })

    # Save the best model locally and log as artifact (same as your original flow)
    model_path = "best_tourism_pred_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face
    repo_id = "affanthinks/Tourism-Package-Prediction"
    repo_type = "model"

    # Step 1: Check if the space exists
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Space '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Space '{repo_id}' not found. Creating new space...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Space '{repo_id}' created.")

    # create_repo("churn-model", repo_type="model", private=False)
    api.upload_file(
        path_or_fileobj="best_tourism_pred_model_v1.joblib",
        path_in_repo="best_tourism_pred_model_v1.joblib",
        repo_id=repo_id,
        repo_type=repo_type,
    )
