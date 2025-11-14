""" 
Script para entrenar un modelo de clasificación utilizando la técnica con mejor rendimiento 
que fuera seleccionada durante la experimentación.
"""
# Importaciones generales
import pandas as pd
import mlflow
import mlflow.sklearn
from pathlib import Path
import joblib
import json
import hashlib
# Importaciones para el preprocesamiento y modelado
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler 
from sklearn.utils import resample
# Importaciones para la evaluación - experimentación
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    f1_score, recall_score, precision_score, 
    accuracy_score, confusion_matrix, roc_auc_score
)
from mlflow.models import infer_signature
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import argparse


def get_data_hash(filepath):
    """
    Calcula el hash MD5 del archivo de datos para trazabilidad.
    
    Parámetros:
        filepath: Ruta al archivo de datos
        
    Retorna:
        Hash MD5 del archivo como string hexadecimal
    """
    with open(filepath, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()


def load_data(path):
    """
    Carga los datos desde un archivo CSV y los divide en conjuntos de entrenamiento y prueba.
    
    Parámetros:
        path: Ruta al archivo CSV con los datos
        
    Retorna:
        X_train, X_test, y_train, y_test: Conjuntos de datos divididos
        
    Excepciones:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si la columna objetivo 'y' no está presente
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"No se encuentra el archivo: {path}")
    
    df = pd.read_csv(path)
    
    if 'y' not in df.columns:
        raise ValueError("El dataset no contiene la columna objetivo 'y'")
    
    print(f"Dataset cargado exitosamente: {len(df)} registros, {len(df.columns)} columnas")
    
    X = df.drop('y', axis=1)
    y = df['y']
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def create_preprocessor(X_train):
    """
    Crea un preprocesador que aplica escalado robusto a variables numéricas
    y codificación one-hot a variables categóricas.
    
    Parámetros:
        X_train: DataFrame con los datos de entrenamiento
        
    Retorna:
        preprocessor: ColumnTransformer configurado
        X_train_converted: DataFrame con columnas convertidas a tipos apropiados
    """
    numerical_columns = X_train.select_dtypes(exclude='object').columns
    categorical_columns = X_train.select_dtypes(include='object').columns

    X_train = X_train.copy()
    int_columns = X_train.select_dtypes(include='int').columns
    for col in int_columns:
        X_train[col] = X_train[col].astype('float')
    
    # Actualizar numerical_cols
    numerical_columns = X_train.select_dtypes(exclude='object').columns

    # Pipeline para valores numéricos
    num_pipeline = Pipeline(steps=[
        ('RobustScaler', RobustScaler())
    ])

    # Pipeline para valores categóricos
    cat_pipeline = Pipeline(steps=[
        ('OneHotEncoder', OneHotEncoder(drop='first', sparse_output=False))
    ])

    # Se configuran los preprocesadores
    preprocessor_full = ColumnTransformer([
        ('num_pipeline', num_pipeline, numerical_columns),
        ('cat_pipeline', cat_pipeline, categorical_columns)
    ]).set_output(transform='pandas')

    return preprocessor_full, X_train


def balance_data(X, y, random_state=42):
    """
    Aplica sub-muestreo (undersampling) para balancear las clases del dataset.
    
    Parámetros:
        X: DataFrame con las características preprocesadas
        y: Serie con las etiquetas
        random_state: Semilla para reproducibilidad
        
    Retorna:
        x_train_resampled: Características balanceadas
        y_train_resampled: Etiquetas balanceadas
    """
    # Combinar los datos preprocesados con las etiquetas
    train_data = X.copy()
    train_data['target'] = y.reset_index(drop=True)

    # Separar por clase
    class_0 = train_data[train_data['target'] == 0]
    class_1 = train_data[train_data['target'] == 1]

    # Encontrar la clase minoritaria
    min_count = min(len(class_0), len(class_1))

    # Submuestreo balanceado - tomar una muestra igual al tamaño de la clase minoritaria
    class_0_balanced = resample(class_0, n_samples=min_count, random_state=random_state)
    class_1_balanced = resample(class_1, n_samples=min_count, random_state=random_state)

    # Combinar las clases balanceadas
    balanced_data = pd.concat([class_0_balanced, class_1_balanced])

    # Separar características y objetivo
    x_train_resampled = balanced_data.drop('target', axis=1)
    y_train_resampled = balanced_data['target']

    return x_train_resampled, y_train_resampled


def train_model(
    data_path: str = 'data/processed/data.csv',
    model_output_path: str = 'models/knn_model',
    preprocessor_output_path: str = 'models/preprocessor.pkl',
    metrics_output_path: str = 'models/metrics.json'
):
    """
    Método principal para entrenar el modelo de clasificación con integración completa de MLflow.
    
    Parámetros:
        data_path: Ruta al archivo de datos procesados
        model_output_path: Ruta donde guardar el modelo entrenado
        preprocessor_output_path: Ruta donde guardar el preprocesador
        metrics_output_path: Ruta donde guardar las métricas en formato JSON
        
    Retorna:
        model: Modelo entrenado
        preprocessor: Preprocesador ajustado
        metrics: Diccionario con las métricas de evaluación
    """
    # Configuración de MLflow
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Proyecto 13MBID-ABR2526 - Producción")

    with mlflow.start_run(run_name="DecisionTree_Production"):
        print("Cargando datos...")
        X_train, X_test, y_train, y_test = load_data(data_path)
        
        # Registrar hash del dataset para trazabilidad
        data_hash = get_data_hash(data_path)
        mlflow.log_param("data_hash", data_hash)
        mlflow.log_param("data_file", data_path)
        print(f"Hash del dataset: {data_hash}")
        
        print("\nCreando preprocesador...")
        preprocessor, X_train_converted = create_preprocessor(X_train)
        X_test = X_test.copy()
        
        # Convertir columnas enteras en X_test también
        int_columns = X_test.select_dtypes(include=['int64', 'int32']).columns
        for col in int_columns:
            X_test[col] = X_test[col].astype('float64')
        
        print("Preprocesando datos...")
        X_train_prep = preprocessor.fit_transform(X_train_converted)
        X_test_prep = preprocessor.transform(X_test)
            
        print("\nBalanceando datos...")
        X_train_balanced, y_train_balanced = balance_data(X_train_prep, y_train)
        
        print(f"  Tamaño original: {len(X_train_prep)}")
        print(f"  Tamaño balanceado: {len(X_train_balanced)}")
        print(f"  Distribución: {y_train_balanced.value_counts().to_dict()}")

        print("\nEntrenando modelo K-Nearest Neighbors...")
        model = KNeighborsClassifier(n_neighbors=7)
        model.fit(X_train_balanced, y_train_balanced)
        
        print("Evaluando modelo...")
        y_pred = model.predict(X_test_prep)
        y_pred_proba = model.predict_proba(X_test_prep)[:, 1]
        
        # Crear pipeline completo
        full_pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        # Signature para el pipeline completo
        pipeline_signature = infer_signature(
            X_train,  # Datos de entrada sin procesar
            y_pred    # Predicciones del modelo
        )
        
        # Signature para el preprocesador
        preprocessor_signature = infer_signature(
            X_train,      # Datos de entrada sin procesar
            X_train_prep  # Datos procesados
        )
        
        # Signature para el modelo
        model_signature = infer_signature(
            X_train_prep,  # Datos procesados
            y_pred         # Predicciones
        )

        # Calcular métricas (incluyendo ROC-AUC)
        metrics = {
            "f1_score": float(f1_score(y_test, y_pred)),
            "recall_score": float(recall_score(y_test, y_pred)),
            "precision_score": float(precision_score(y_test, y_pred)),
            "accuracy_score": float(accuracy_score(y_test, y_pred)),
            "roc_auc_score": float(roc_auc_score(y_test, y_pred_proba))
        }
        
        print("\nMétricas del modelo:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)

        # Registrar parámetros
        mlflow.log_params({
            "model_type": "KNeighborsClassifier",
            "leaf_size": model.leaf_size,
            "metric": model.metric,
            "metric_params": model.metric_params,
            "n_jobs": model.n_jobs,
            "n_neighbors": model.n_neighbors,
            "p": model.p,
            "weights": model.weights,
            "balancing_method": "undersampling",
            "train_samples": len(X_train_balanced),
            "test_samples": len(X_test),
            "random_state": 42
        })

        
        # Registrar métricas
        mlflow.log_metrics(metrics)

        # Registrar matriz de confusión      
        fig, ax = plt.subplots(figsize=(8, 6))
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No', 'Yes']).plot(ax=ax)
        plt.title('Confusion Matrix - Production Model')
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close()

        # Registrar pipeline completo
        mlflow.sklearn.log_model(
            sk_model=full_pipeline,
            artifact_path="model",
            signature=pipeline_signature,
        )
        
        # Registrar preprocesador
        mlflow.sklearn.log_model(
            sk_model=preprocessor,
            artifact_path="preprocessor",
            signature=preprocessor_signature,
        )
        
        # Registrar modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="classifier",
            signature=model_signature,
        )

        # Guardar modelos localmente
        print("\nGuardando modelos...")
        Path(model_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(preprocessor_output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(metrics_output_path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(model, model_output_path)
        joblib.dump(preprocessor, preprocessor_output_path)
        
        # Guardar métricas
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\nModelo guardado en: {model_output_path}")
        print(f"Preprocesador guardado en: {preprocessor_output_path}")
        print(f"Métricas guardadas en: {metrics_output_path}")

        return model, preprocessor, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar modelo de producción")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/processed/bank-processed.csv",
        help="Ruta al archivo de datos procesados"
    )
    parser.add_argument(
        "--model-output",
        type=str,
        default="models/knn_model.pkl",
        help="Ruta donde guardar el modelo"
    )
    parser.add_argument(
        "--preprocessor-output",
        type=str,
        default="models/preprocessor.pkl",
        help="Ruta donde guardar el preprocesador"
    )
    parser.add_argument(
        "--metrics-output",
        type=str,
        default="metrics/metrics.json",
        help="Ruta donde guardar las métricas"
    )
    
    args = parser.parse_args()
    
    train_model(
        data_path=args.data_path,
        model_output_path=args.model_output,
        preprocessor_output_path=args.preprocessor_output,
        metrics_output_path=args.metrics_output
    )