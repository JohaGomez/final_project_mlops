import os
import zipfile
import urllib.request
import pandas as pd
from sklearn.model_selection import train_test_split
from utils import load_config, ensure_dir


def download_dataset(url: str, dest_folder: str) -> str:
    """Descarga y descomprime el dataset UCI."""
    ensure_dir(dest_folder)
    zip_path = os.path.join(dest_folder, "bank.zip")

    if not os.path.exists(zip_path):
        print(f"📥 Descargando dataset desde: {url}")
        urllib.request.urlretrieve(url, zip_path)
    else:
        print("✅ Dataset ya descargado.")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dest_folder)

    # Ruta del CSV principal
    csv_path = os.path.join(dest_folder, "bank-additional", "bank-additional-full.csv")
    return csv_path


def preprocess_data(cfg: dict) -> tuple:
    """Carga, limpia, codifica y divide el dataset."""
    csv_path = download_dataset(cfg["data"]["external_url"], cfg["data"]["raw_dir"])
    df = pd.read_csv(csv_path, sep=";")

    print(f"✅ Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Eliminar nulos
    df = df.dropna()

    # Codificar variables categóricas
    df = pd.get_dummies(df, drop_first=True)

    # Dividir
    X = df.drop(columns=[f"{cfg['data']['target']}_yes"])
    y = df[f"{cfg['data']['target']}_yes"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["test_size"], random_state=cfg["random_state"], stratify=y
    )

    # Guardar procesados
    ensure_dir(cfg["data"]["processed_dir"])
    X_train.to_csv(os.path.join(cfg["data"]["processed_dir"], "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(cfg["data"]["processed_dir"], "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(cfg["data"]["processed_dir"], "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(cfg["data"]["processed_dir"], "y_test.csv"), index=False)

    print("💾 Datos procesados guardados en data/processed/")
    return X_train, X_test, y_train, y_test


def main():
    cfg = load_config("configs/config.yaml")
    preprocess_data(cfg)


if __name__ == "__main__":
    main()
