import argparse
import os
import pandas as pd
import mlflow
from mlflow.models.signature import infer_signature
from mlflow import sklearn as mlflow_sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from utils import load_config, ensure_dir


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config.yaml")
    return p.parse_args()


def load_processed_data(cfg):
    """Carga los datos procesados desde data/processed/"""
    data_dir = cfg["data"]["processed_dir"]
    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Configuración MLflow
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    ensure_dir("models")

    # Cargar datos procesados
    X_train, X_test, y_train, y_test = load_processed_data(cfg)

    # Inicializar modelo
    model_params = cfg["model"]["params"]
    model = LogisticRegression(**model_params)

    with mlflow.start_run(run_name="logistic_regression_training"):
        # Entrenar
        model.fit(X_train, y_train)

        # Predicciones y métricas
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Logging en MLflow
        mlflow.log_params(model_params)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Guardar modelo con firma e input_example
        signature = infer_signature(X_train, model.predict(X_train))
        input_example = X_train.head(2)
        mlflow_sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
            registered_model_name=None,
        )

        # Guardar localmente también
        model_path = os.path.join("models", "logreg_model.pkl")
        import joblib
        joblib.dump(model, model_path)

        print(f"✅ Modelo entrenado. Accuracy: {acc:.4f} | F1: {f1:.4f}")
        print(f"💾 Modelo guardado en: {model_path}")

    print("📊 Run registrado en MLflow. Revisa http://localhost:5000 si abres el UI.")


if __name__ == "__main__":
    main()
