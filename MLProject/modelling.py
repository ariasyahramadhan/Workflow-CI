import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--n_estimators', type=int, default=100)
parser.add_argument('--max_depth', type=int, default=5)
parser.add_argument('--min_samples_split', type=int, default=2)
args = parser.parse_args()

# Load Data
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'titanic_preprocessing')

X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
X_test  = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).squeeze()
y_test  = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).squeeze()

print(f"Train: {X_train.shape} | Test: {X_test.shape}")

# Training
model = RandomForestClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    min_samples_split=args.min_samples_split,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc  = accuracy_score(y_test, y_pred)
f1   = f1_score(y_test, y_pred, average='weighted')
prec = precision_score(y_test, y_pred, average='weighted')
rec  = recall_score(y_test, y_pred, average='weighted')
auc  = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")

# MLflow Logging
with mlflow.start_run():
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision_score", prec)
    mlflow.log_metric("recall_score", rec)
    mlflow.log_metric("roc_auc", auc)

    # Artefak 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Tidak Selamat', 'Selamat'],
                yticklabels=['Tidak Selamat', 'Selamat'])
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    fig.savefig('confusion_matrix.png', dpi=100)
    mlflow.log_artifact('confusion_matrix.png')
    plt.close()

    # Artefak 2: Feature Importance
    importances = pd.Series(
        model.feature_importances_,
        index=X_train.columns
    ).sort_values()
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    importances.plot(kind='barh', ax=ax2, color='#534AB7')
    ax2.set_title('Feature Importances')
    plt.tight_layout()
    fig2.savefig('feature_importance.png', dpi=100)
    mlflow.log_artifact('feature_importance.png')
    plt.close()

    mlflow.sklearn.log_model(model, "model")
    print(f"Run ID: {mlflow.active_run().info.run_id}")

print("Training selesai!")