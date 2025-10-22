# README – Automatización de un Pipeline de Machine Learning con GitHub Actions

## 1. Objetivo General
Desarrollar un pipeline reproducible de machine learning que permita entrenar, evaluar y registrar un modelo, aplicando principios de MLOps y utilizando un flujo de integración y entrega continua (CI/CD) automatizado mediante GitHub Actions.

---

## 2. Descripción del Proyecto
Este proyecto implementa un pipeline de Machine Learning completamente automatizado, que integra las fases de:
1. Carga y procesamiento de datos  
2. Entrenamiento y evaluación del modelo  
3. Registro de métricas y artefactos en MLflow  
4. Ejecución automatizada del flujo mediante GitHub Actions  

El objetivo es garantizar la reproducibilidad, trazabilidad y calidad del modelo en un entorno controlado de CI/CD.

---

## 3. Dataset Utilizado
**Fuente:** [UCI Machine Learning Repository – Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)  
**Tamaño:** 45.211 registros, 17 atributos  
**Tipo:** Clasificación binaria (`yes` / `no`)  
**Objetivo:** Predecir si un cliente contratará un depósito a plazo.

El dataset fue seleccionado por su documentación académica y su uso frecuente en estudios de clasificación supervisada.

---

## 4. Preprocesamiento de Datos
Las transformaciones incluyeron:
- Manejo de valores nulos y tipos de datos  
- Codificación de variables categóricas con *One-Hot Encoding*  
- Escalamiento de características numéricas con `StandardScaler`  
- División del dataset en entrenamiento (80%) y prueba (20%)

Los datos procesados se almacenan en `data/processed/`.

---

## 5. Entrenamiento y Evaluación del Modelo
- **Modelo:** Regresión Logística (`LogisticRegression` de scikit-learn)
- **Hiperparámetros:**  
  - `C = 1.0`, `solver = 'lbfgs'`, `max_iter = 100`
- **Métricas:**  
  - `Accuracy = 0.91`  
  - `F1-score = 0.51`

El modelo fue evaluado con métricas relevantes para problemas de clasificación binaria y se registró en MLflow.

---

## 6. Tracking con MLflow
Se utilizó **MLflow Tracking** para registrar:
- Parámetros (`log_param`)
- Métricas (`log_metric`)
- Modelo y artefactos (`log_model`)
- Firma del modelo (`signature`)
- Ejemplo de entrada (`input_example`)

El tracking se ejecuta en modo local (`file://.../mlruns`) y permite visualizar los experimentos con el comando:
```bash
make mlflow-ui

## 6. Evidencia del modelo registrado con MLflow

El siguiente run corresponde al entrenamiento y registro del modelo de regresión logística
dentro del experimento **ci-cd-mlflow-local**, ejecutado automáticamente desde el pipeline CI/CD.

- **Run ID:** 756bf6dce24544799f0ca69b7319f357  
- **Accuracy:** 0.9129  
- **F1 Score:** 0.5132  
- **Archivo fuente:** `src/train.py`  
- **Estado:** Finished ✅  