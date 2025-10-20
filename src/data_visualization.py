"""
Script de exploración de datos
================================

Este script realiza un análisis exploratorio básico de los datos
y genera visualizaciones que se guardan en archivos.

Funciones principales:
- Cargar datos desde CSV
- Mostrar estadísticas básicas
- Crear gráficos de distribuciones
- Generar matriz de correlación
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings

# Configurar para que no muestre advertencias molestas
warnings.filterwarnings('ignore')

# Configurar estilo de los gráficos
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (8, 5)


def cargar_datos(ruta_archivo, separador=';'):
    """
    Carga un archivo CSV y lo convierte en un DataFrame.
    
    Parámetros:
    -----------
    ruta_archivo : str
        Ruta completa al archivo CSV que queremos leer
    separador : str
        Carácter que separa las columnas (por defecto es punto y coma)
    
    Retorna:
    --------
    DataFrame con los datos cargados
    
    Ejemplo:
    --------
    df = cargar_datos('data/raw/bank-additional-full.csv')
    """
    print(f"Cargando datos desde: {ruta_archivo}")
    df = pd.read_csv(ruta_archivo, sep=separador)
    print(f"Datos cargados exitosamente: {df.shape[0]} filas y {df.shape[1]} columnas")
    return df


def mostrar_informacion_basica(df):
    """
    Muestra información básica del DataFrame:
    - Dimensiones (filas y columnas)
    - Tipos de datos
    - Valores nulos
    
    Parámetros:
    -----------
    df : DataFrame
        El DataFrame que queremos analizar
    """
    print("\n" + "="*70)
    print("INFORMACIÓN BÁSICA DEL DATASET")
    print("="*70)
    
    # Mostrar dimensiones
    print(f"\nDimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    
    # Mostrar primeras filas
    print("\nPrimeras 5 filas:")
    print(df.head())
    
    # Información de tipos de datos
    print("\nInformación de columnas:")
    df.info()
    
    # Estadísticas de variables numéricas
    print("\nEstadísticas de variables numéricas:")
    print(df.describe())
    
    # Estadísticas de variables categóricas
    print("\nEstadísticas de variables categóricas:")
    print(df.describe(include='object'))


def crear_carpeta_salida(ruta_carpeta):
    """
    Crea una carpeta para guardar los gráficos si no existe.
    
    Parámetros:
    -----------
    ruta_carpeta : str
        Ruta de la carpeta que queremos crear
    """
    if not os.path.exists(ruta_carpeta):
        os.makedirs(ruta_carpeta)
        print(f"Carpeta creada: {ruta_carpeta}")
    else:
        print(f"Carpeta ya existe: {ruta_carpeta}")


def graficar_variable_objetivo(df, columna_objetivo, carpeta_salida):
    """
    Crea un gráfico de barras para la variable objetivo.
    Muestra cuántos casos hay de cada clase.
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame con los datos
    columna_objetivo : str
        Nombre de la columna objetivo
    carpeta_salida : str
        Carpeta donde guardar el gráfico
    """
    print("\nGenerando gráfico de variable objetivo...")
    
    # Crear el gráfico
    plt.figure(figsize=(8, 5))
    sns.countplot(x=columna_objetivo, data=df)
    plt.title(f"Distribución de {columna_objetivo}")
    plt.xlabel(columna_objetivo)
    plt.ylabel("Frecuencia")
    
    # Guardar el gráfico
    nombre_archivo = os.path.join(carpeta_salida, "01_variable_objetivo.png")
    plt.savefig(nombre_archivo, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Gráfico guardado: {nombre_archivo}")
    
    # Mostrar porcentajes
    print("\nDistribución porcentual:")
    porcentajes = df[columna_objetivo].value_counts(normalize=True).mul(100).round(2)
    print(porcentajes)


def graficar_variables_categoricas(df, columna_objetivo, carpeta_salida):
    """
    Crea gráficos de barras para todas las variables categóricas
    (excepto la variable objetivo).
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame con los datos
    columna_objetivo : str
        Nombre de la columna objetivo para excluirla
    carpeta_salida : str
        Carpeta donde guardar los gráficos
    """
    print("\nGenerando gráficos de variables categóricas...")
    
    # Obtener todas las columnas categóricas excepto la objetivo
    columnas_categoricas = df.select_dtypes(include=["object"]).columns
    columnas_categoricas = [col for col in columnas_categoricas if col != columna_objetivo]
    
    # Crear un gráfico para cada variable categórica
    for i, columna in enumerate(columnas_categoricas):
        plt.figure(figsize=(8, 4))
        
        # Ordenar las categorías por frecuencia
        orden = df[columna].value_counts().index
        sns.countplot(y=columna, data=df, order=orden)
        
        plt.title(f"Distribución de {columna}")
        plt.xlabel("Cantidad")
        plt.ylabel(columna)
        
        # Guardar el gráfico
        nombre_archivo = os.path.join(carpeta_salida, f"02_categorica_{columna}.png")
        plt.savefig(nombre_archivo, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Gráfico {i+1}/{len(columnas_categoricas)}: {columna}")
    
    print(f"Total de gráficos categóricos creados: {len(columnas_categoricas)}")


def graficar_matriz_correlacion(df, carpeta_salida):
    """
    Crea un mapa de calor (heatmap) con las correlaciones
    entre todas las variables numéricas.
    
    Parámetros:
    -----------
    df : DataFrame
        DataFrame con los datos
    carpeta_salida : str
        Carpeta donde guardar el gráfico
    """
    print("\nGenerando matriz de correlación...")
    
    # Seleccionar solo columnas numéricas
    df_numerico = df.select_dtypes(include=['float64', 'int64'])
    
    # Calcular correlaciones
    correlacion = df_numerico.corr()
    
    # Crear el gráfico
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlacion, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Matriz de correlaciones')
    
    # Guardar el gráfico
    nombre_archivo = os.path.join(carpeta_salida, "03_matriz_correlacion.png")
    plt.savefig(nombre_archivo, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Matriz de correlación guardada: {nombre_archivo}")


def comparar_categoricas_con_objetivo(df, columna_objetivo, carpeta_salida):
    """
    Compara cada variable categórica con la variable objetivo.
    Crea gráficos de barras agrupadas que muestran la relación
    entre cada categoría y los valores de la variable objetivo.
       
    Parámetros:
    -----------
    df : DataFrame
        DataFrame con los datos
    columna_objetivo : str
        Nombre de la columna objetivo
    carpeta_salida : str
        Carpeta donde guardar los gráficos
    """
    print("\nGenerando gráficos de comparación con variable objetivo...")
    
    # Obtener todas las columnas categóricas excepto la objetivo
    columnas_categoricas = df.select_dtypes(include=["object"]).columns
    columnas_categoricas = [col for col in columnas_categoricas if col != columna_objetivo]
    
    # Crear un gráfico comparativo para cada variable categórica
    for i, columna in enumerate(columnas_categoricas):
        # Crear figura más grande para mejor visualización
        plt.figure(figsize=(10, 6))
        
        # Crear tabla cruzada (crosstab) con porcentajes
        tabla_cruzada = pd.crosstab(df[columna], df[columna_objetivo], normalize='index') * 100
        
        # Crear gráfico de barras agrupadas
        tabla_cruzada.plot(kind='bar', stacked=False)
        
        plt.title(f"Relación entre {columna} y variable objetivo")
        plt.xlabel(columna)
        plt.ylabel("Porcentaje (%)")
        plt.legend(title=columna_objetivo)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Guardar el gráfico
        nombre_archivo = os.path.join(carpeta_salida, f"04_comparacion_{columna}_vs_{columna_objetivo}.png")
        plt.savefig(nombre_archivo, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Comparación {i+1}/{len(columnas_categoricas)}: {columna} vs {columna_objetivo}")
    
    print(f"Total de gráficos de comparación creados: {len(columnas_categoricas)}")

def comparar_numericas_con_objetivo(df, columna_objetivo, carpeta_salida):
    """
    Compara cada variable numérica con la variable objetivo.
    Crea gráficos de distribución (boxplot y violin) que muestran
    cómo se distribuyen los valores numéricos según la variable objetivo.
        
    Parámetros:
    -----------
    df : DataFrame
        DataFrame con los datos
    columna_objetivo : str
        Nombre de la columna objetivo
    carpeta_salida : str
        Carpeta donde guardar los gráficos
    """
    print("\nGenerando gráficos de comparación de variables numéricas...")
    
    # Obtener todas las columnas numéricas
    columnas_numericas = df.select_dtypes(include=['float64', 'int64']).columns
    
    # Crear gráficos para cada variable numérica
    for i, columna in enumerate(columnas_numericas):
        # Crear figura con dos subgráficos (boxplot y violin)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Gráfico 1: Boxplot
        sns.boxplot(x=columna_objetivo, y=columna, data=df, ax=axes[0])
        axes[0].set_title(f"Boxplot: {columna} por {columna_objetivo}")
        axes[0].set_xlabel(columna_objetivo)
        axes[0].set_ylabel(columna)
        
        # Gráfico 2: Violin plot
        sns.violinplot(x=columna_objetivo, y=columna, data=df, ax=axes[1])
        axes[1].set_title(f"Violin: {columna} por {columna_objetivo}")
        axes[1].set_xlabel(columna_objetivo)
        axes[1].set_ylabel(columna)
        
        plt.tight_layout()
        
        # Guardar el gráfico
        nombre_archivo = os.path.join(carpeta_salida, f"05_comparacion_numerica_{columna}_vs_{columna_objetivo}.png")
        plt.savefig(nombre_archivo, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"Comparación numérica {i+1}/{len(columnas_numericas)}: {columna} vs {columna_objetivo}")
    
    print(f"Total de gráficos numéricos de comparación creados: {len(columnas_numericas)}")


def explorar_datos(ruta_datos, columna_objetivo='y', carpeta_salida='docs/figures'):
    """
    Función principal que ejecuta todo el análisis exploratorio.
    
    Esta función:
    1. Carga los datos
    2. Muestra información básica
    3. Crea todos los gráficos
    4. Guarda las visualizaciones
    
    Parámetros:
    -----------
    ruta_datos : str
        Ruta al archivo CSV con los datos
    columna_objetivo : str
        Nombre de la columna objetivo (por defecto 'y')
    carpeta_salida : str
        Carpeta donde guardar los gráficos
    """
    print("\n" + "="*70)
    print("INICIANDO EXPLORACIÓN DE DATOS")
    print("="*70)
    
    # Paso 1: Cargar los datos
    df = cargar_datos(ruta_datos)
    
    # Paso 2: Crear carpeta para guardar gráficos
    crear_carpeta_salida(carpeta_salida)
    
    # Paso 3: Mostrar información básica
    mostrar_informacion_basica(df)
    
    # Paso 4: Crear gráfico de variable objetivo
    graficar_variable_objetivo(df, columna_objetivo, carpeta_salida)
    
    # Paso 5: Crear gráficos de variables categóricas
    graficar_variables_categoricas(df, columna_objetivo, carpeta_salida)
    
    # Paso 6: Crear matriz de correlación
    graficar_matriz_correlacion(df, carpeta_salida)

    # Paso 7: Comparar variables categóricas con variable objetivo
    comparar_categoricas_con_objetivo(df, columna_objetivo, carpeta_salida)

    # Paso 8: Comparar variables numéricas con variable objetivo
    comparar_numericas_con_objetivo(df, columna_objetivo, carpeta_salida)
    
    print("\n" + "="*70)
    print("EXPLORACIÓN COMPLETADA")
    print("="*70)
    print(f"Los gráficos se guardaron en: {carpeta_salida}")


if __name__ == "__main__":
    """
    Este bloque se ejecuta cuando corremos el script directamente.
    Aquí definimos la ruta de los datos y ejecutamos la exploración.
    """
    
    # Configuración: cambiar estas rutas según tu proyecto
    RUTA_DATOS = '../data/raw/bank-additional-full.csv'
    CARPETA_GRAFICOS = '../docs/figures'
    COLUMNA_OBJETIVO = 'y'
    
    # Ejecutar la exploración
    explorar_datos(
        ruta_datos=RUTA_DATOS,
        columna_objetivo=COLUMNA_OBJETIVO,
        carpeta_salida=CARPETA_GRAFICOS
    )