"""
Script de preparación de datos para el proyecto de marketing bancario.

Este script realiza las siguientes transformaciones:
1. Selección de datos: eliminación de columnas con baja calidad
2. Limpieza: eliminación de duplicados y valores nulos
3. Construcción: creación de nuevas variables derivadas
4. Formateo: normalización y transformación de datos
"""

import pandas as pd
import numpy as np
from pathlib import Path


# Configuración de rutas
INPUT_CSV = 'data/raw/bank-additional-full.csv'
OUTPUT_CSV = 'data/processed/bank-processed.csv'
DOCS_PATH = 'docs/transformations.txt'


def load_data(input_path=INPUT_CSV):
    """
    Carga el dataset desde el archivo CSV.
    
    Args:
        input_path (str): Ruta al archivo CSV de entrada
        
    Returns:
        pd.DataFrame: DataFrame con los datos cargados
    """
    print(f"Cargando datos desde {input_path}...")
    df = pd.read_csv(input_path, sep=';')
    print(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def normalize_column_names(df):
    """
    Normaliza los nombres de las columnas reemplazando puntos por guiones bajos.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con columnas normalizadas
    """
    print("Normalizando nombres de columnas...")
    df.columns = df.columns.str.replace(".", "_", regex=False)
    return df


def handle_unknown_values(df):
    """
    Transforma los valores 'unknown' a NaN para facilitar el tratamiento.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con valores 'unknown' convertidos a NaN
    """
    print("Reemplazando valores 'unknown' por NaN...")
    unknown_count = (df == 'unknown').sum().sum()
    df.replace('unknown', np.nan, inplace=True)
    print(f"Se reemplazaron {unknown_count} valores 'unknown'")
    return df


def select_relevant_columns(df):
    """
    Elimina columnas con baja calidad de datos.
    
    Criterios de eliminación:
    - 'default': 99.9% de una sola clase + >20% valores nulos
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame sin columnas de baja calidad
    """
    print("Eliminando columnas con baja calidad de datos...")
    columns_to_drop = ['default']
    
    for col in columns_to_drop:
        if col in df.columns:
            print(f"  - Eliminando columna '{col}' (baja variedad y muchos nulos)")
            df.drop(columns=[col], inplace=True)
    
    return df


def remove_null_rows(df):
    """
    Elimina las filas que contienen valores nulos.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame sin filas nulas
    """
    print("Eliminando filas con valores nulos...")
    rows_before = df.shape[0]
    df.dropna(inplace=True)
    rows_after = df.shape[0]
    rows_removed = rows_before - rows_after
    print(f"Se eliminaron {rows_removed} filas con valores nulos")
    return df


def remove_duplicates(df):
    """
    Elimina las filas duplicadas del dataset.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame sin duplicados
    """
    print("Eliminando filas duplicadas...")
    rows_before = df.shape[0]
    df.drop_duplicates(inplace=True)
    rows_after = df.shape[0]
    duplicates_removed = rows_before - rows_after
    print(f"Se eliminaron {duplicates_removed} filas duplicadas")
    return df


def create_previous_contact_feature(df):
    """
    Crea una variable binaria para indicar si hubo contacto previo.
    
    La variable 'had_previous_contact' se crea basándose en 'pdays':
    - 1: Si pdays != 999 (hubo contacto previo)
    - 0: Si pdays == 999 (no hubo contacto previo)
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con la nueva variable
    """
    print("Creando variable 'had_previous_contact'...")
    if 'pdays' in df.columns:
        df['had_previous_contact'] = (df['pdays'] != 999).astype(int)
        print(f"  - Contactos previos: {df['had_previous_contact'].sum()}")
        print(f"  - Sin contacto previo: {(df['had_previous_contact'] == 0).sum()}")
    return df


def remove_redundant_columns(df):
    """
    Elimina columnas redundantes después de la creación de variables derivadas.
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame sin columnas redundantes
    """
    print("Eliminando columnas redundantes...")
    if 'pdays' in df.columns:
        print("  - Eliminando 'pdays' (reemplazada por 'had_previous_contact')")
        df.drop(columns=['pdays'], inplace=True)
    return df


def encode_target_variable(df):
    """
    Codifica la variable objetivo 'y' a valores binarios.
    
    Mapeo:
    - 'yes' -> 1
    - 'no' -> 0
    
    Args:
        df (pd.DataFrame): DataFrame original
        
    Returns:
        pd.DataFrame: DataFrame con variable objetivo codificada
    """
    print("Codificando variable objetivo 'y'...")
    target_map = {'yes': 1, 'no': 0}
    df['y'] = df['y'].map(target_map)
    print(f"  - Clase positiva (y=1): {(df['y'] == 1).sum()} ({(df['y'] == 1).sum()/len(df)*100:.2f}%)")
    print(f"  - Clase negativa (y=0): {(df['y'] == 0).sum()} ({(df['y'] == 0).sum()/len(df)*100:.2f}%)")
    return df


def save_processed_data(df, output_path=OUTPUT_CSV):
    """
    Guarda el dataset procesado en un archivo CSV.
    
    Args:
        df (pd.DataFrame): DataFrame procesado
        output_path (str): Ruta de salida para el archivo CSV
    """
    print(f"\nGuardando datos procesados en {output_path}...")
    
    # Crear directorio si no existe
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_path, index=False)
    print(f"Datos guardados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")


def save_transformation_log(initial_shape, final_shape, docs_path=DOCS_PATH):
    """
    Guarda un registro de las transformaciones realizadas.
    
    Args:
        initial_shape (tuple): Dimensiones iniciales (filas, columnas)
        final_shape (tuple): Dimensiones finales (filas, columnas)
        docs_path (str): Ruta del archivo de documentación
    """
    print(f"\nGenerando log de transformaciones en {docs_path}...")
    
    # Crear directorio si no existe
    docs_dir = Path(docs_path).parent
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    with open(docs_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("REGISTRO DE TRANSFORMACIONES DEL DATASET\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("1. SELECCIÓN DE DATOS\n")
        f.write("-" * 70 + "\n")
        f.write("   - Se eliminó la columna 'default':\n")
        f.write("     * Razón: Baja variedad (99.9% de una sola clase)\n")
        f.write("     * Razón: Alta proporción de valores nulos (>20%)\n\n")
        
        f.write("2. LIMPIEZA DE DATOS\n")
        f.write("-" * 70 + "\n")
        f.write("   - Se reemplazaron valores 'unknown' por NaN\n")
        f.write(f"   - Se eliminaron {initial_shape[0] - final_shape[0]} filas:\n")
        f.write("     * Filas con valores nulos\n")
        f.write("     * Filas duplicadas\n\n")
        
        f.write("3. CONSTRUCCIÓN DE DATOS\n")
        f.write("-" * 70 + "\n")
        f.write("   - Se creó la variable 'had_previous_contact':\n")
        f.write("     * Binarización basada en 'pdays'\n")
        f.write("     * 1: Hubo contacto previo (pdays != 999)\n")
        f.write("     * 0: No hubo contacto previo (pdays == 999)\n")
        f.write("   - Se eliminó la columna 'pdays' (reemplazada por la nueva variable)\n\n")
        
        f.write("4. FORMATEO DE DATOS\n")
        f.write("-" * 70 + "\n")
        f.write("   - Se normalizaron los nombres de columnas (puntos por guiones bajos)\n")
        f.write("   - Se codificó la variable objetivo 'y':\n")
        f.write("     * 'yes' -> 1\n")
        f.write("     * 'no' -> 0\n\n")
        
        f.write("=" * 70 + "\n")
        f.write("RESUMEN\n")
        f.write("=" * 70 + "\n")
        f.write(f"Dataset original:  {initial_shape[0]:,} filas × {initial_shape[1]} columnas\n")
        f.write(f"Dataset procesado: {final_shape[0]:,} filas × {final_shape[1]} columnas\n")
        f.write(f"Filas eliminadas:  {initial_shape[0] - final_shape[0]:,} ({(initial_shape[0] - final_shape[0])/initial_shape[0]*100:.2f}%)\n")
        f.write(f"Columnas eliminadas: {initial_shape[1] - final_shape[1]}\n")
        f.write("=" * 70 + "\n")
    
    print("Log de transformaciones guardado exitosamente")


def preprocess_data(input_path=INPUT_CSV, output_path=OUTPUT_CSV, docs_path=DOCS_PATH):
    """
    Ejecuta el pipeline completo de preprocesamiento de datos.
    
    Pipeline:
    1. Carga de datos
    2. Normalización de nombres de columnas
    3. Tratamiento de valores 'unknown'
    4. Selección de columnas relevantes
    5. Eliminación de filas nulas
    6. Eliminación de duplicados
    7. Creación de variables derivadas
    8. Eliminación de columnas redundantes
    9. Codificación de variable objetivo
    10. Guardado de datos procesados
    11. Generación de log de transformaciones
    
    Args:
        input_path (str): Ruta al archivo CSV de entrada
        output_path (str): Ruta al archivo CSV de salida
        docs_path (str): Ruta al archivo de documentación
        
    Returns:
        tuple: Dimensiones iniciales y finales del dataset (initial_shape, final_shape)
    """
    print("=" * 70)
    print("INICIANDO PIPELINE DE PREPROCESAMIENTO DE DATOS")
    print("=" * 70 + "\n")
    
    # Cargar datos
    df = load_data(input_path)
    initial_shape = df.shape
    
    # Aplicar transformaciones en orden
    df = normalize_column_names(df)
    df = handle_unknown_values(df)
    df = select_relevant_columns(df)
    df = remove_null_rows(df)
    df = remove_duplicates(df)
    df = create_previous_contact_feature(df)
    df = remove_redundant_columns(df)
    df = encode_target_variable(df)
    
    final_shape = df.shape
    
    # Guardar resultados
    save_processed_data(df, output_path)
    save_transformation_log(initial_shape, final_shape, docs_path)
    
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"\nResumen:")
    print(f"  Dataset original:  {initial_shape[0]:,} filas × {initial_shape[1]} columnas")
    print(f"  Dataset procesado: {final_shape[0]:,} filas × {final_shape[1]} columnas")
    print(f"  Reducción:         {initial_shape[0] - final_shape[0]:,} filas ({(initial_shape[0] - final_shape[0])/initial_shape[0]*100:.2f}%)")
    
    return initial_shape, final_shape


if __name__ == "__main__":
    # Ejecutar pipeline de preprocesamiento
    preprocess_data()