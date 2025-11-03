"""
Test de Calidad de Datos usando pytest

Este script implementa tests automatizados para evaluar la calidad del dataset
usando pytest, la herramienta estándar para testing en Python.
"""

import pytest
import pandas as pd
import numpy as np


# Fixture de pytest para cargar los datos una sola vez
@pytest.fixture(scope="module")
def dataset():
    """
    Carga el dataset como un DataFrame de pandas
    """
    ruta_archivo = 'data/raw/bank-additional-full.csv'
    df = pd.read_csv(ruta_archivo, sep=';')
    return df


def test_columnas_esperadas(dataset):
    """
    Test 1: Verificar que existen las columnas esperadas en el dataset
    """
    columnas_esperadas = [
        'age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
        'contact', 'month', 'day_of_week', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'emp.var.rate', 'cons.price.idx',
        'cons.conf.idx', 'euribor3m', 'nr.employed', 'y'
    ]
    
    # Verificar que el dataset tiene las columnas esperadas
    columnas_actuales = list(dataset.columns)
    
    assert columnas_actuales == columnas_esperadas, f"Las columnas no coinciden. Esperadas: {columnas_esperadas}, Actuales: {columnas_actuales}"


def test_esquema_dataframe(dataset):
    """
    Test 2: Validar el esquema completo del dataframe (tipos de datos esperados)
    """
    # Definir el esquema esperado: {nombre_columna: tipo_esperado}
    esquema_esperado = {
        'age': 'int64',
        'job': 'object',
        'marital': 'object',
        'education': 'object',
        'default': 'object',
        'housing': 'object',
        'loan': 'object',
        'contact': 'object',
        'month': 'object',
        'day_of_week': 'object',
        'duration': 'int64',
        'campaign': 'int64',
        'pdays': 'int64',
        'previous': 'int64',
        'poutcome': 'object',
        'emp.var.rate': 'float64',
        'cons.price.idx': 'float64',
        'cons.conf.idx': 'float64',
        'euribor3m': 'float64',
        'nr.employed': 'float64',
        'y': 'object'
    }
    
    print("\n" + "="*80)
    print("VALIDACIÓN DE ESQUEMA DEL DATAFRAME")
    print("="*80)
    
    errores = []
    
    for columna, tipo_esperado in esquema_esperado.items():
        if columna not in dataset.columns:
            errores.append(f" Columna '{columna}' no existe en el dataset")
            print(f" Columna '{columna}' no existe en el dataset")
        else:
            tipo_actual = str(dataset[columna].dtype)
            if tipo_actual == tipo_esperado:
                print(f" '{columna}': {tipo_actual}")
            else:
                errores.append(f" Columna '{columna}': esperado {tipo_esperado}, encontrado {tipo_actual}")
                print(f" '{columna}': esperado {tipo_esperado}, encontrado {tipo_actual}")
    
    print("="*80)
    
    # Verificar si hay columnas adicionales no esperadas
    columnas_extra = set(dataset.columns) - set(esquema_esperado.keys())
    if columnas_extra:
        print(f"\nColumnas adicionales no esperadas: {columnas_extra}")
        errores.append(f"Columnas adicionales: {columnas_extra}")
    
    # Si hay errores, el test falla
    assert len(errores) == 0, f"\nErrores encontrados en el esquema:\n" + "\n".join(errores)


def test_no_filas_vacias(dataset):
    """
    Test 3: Verificar que el dataset no está vacío
    """
    n_filas = len(dataset)
    
    print(f"\nNúmero de filas en el dataset: {n_filas}")
    
    # Esperamos al menos 1000 filas y menos de 50000
    assert 1000 <= n_filas <= 50000, f"El dataset no tiene un número razonable de filas: {n_filas}"


def test_valores_nulos_todas_columnas(dataset):
    """
    Test 4: Verificar el porcentaje de valores nulos en todas las columnas
    Identificar columnas que excedan el umbral del 20%
    """
    umbral = 20  # Umbral del 20% según la memoria de trabajo
    
    columnas_problematicas = []
    
    for columna in dataset.columns:
        # Calcular el porcentaje de valores nulos
        n_nulos = dataset[columna].isnull().sum()
        porcentaje_nulos = (n_nulos / len(dataset)) * 100
        
        if porcentaje_nulos > 0:
            print(f"\nColumna '{columna}': {n_nulos} valores nulos ({porcentaje_nulos:.2f}%)")
        
        # Si excede el umbral, agregar a la lista de columnas problemáticas
        if porcentaje_nulos > umbral:
            columnas_problematicas.append({
                'columna': columna,
                'porcentaje': porcentaje_nulos
            })
            print(f"  ALERTA: Excede el umbral del {umbral}%")
    
    # Si hay columnas problemáticas, imprimir resumen
    if columnas_problematicas:
        print(f"\n{'='*80}")
        print(f"RESUMEN: {len(columnas_problematicas)} columna(s) exceden el umbral del {umbral}%")
        for item in columnas_problematicas:
            print(f"  - {item['columna']}: {item['porcentaje']:.2f}%")
        print(f"{'='*80}")
        print("Recomendación: Considerar eliminar estas columnas")


def test_valores_unknown_en_categoricas(dataset):
    """
    Test 5: Detectar valores 'unknown' en variables categóricas
    Los valores 'unknown' se consideran como valores faltantes
    """
    columnas_categoricas = ['job', 'marital', 'education', 'default', 
                           'housing', 'loan', 'contact', 'poutcome']
    
    umbral = 15  # Umbral del 15% para valores 'unknown'
    columnas_con_unknown = []
    
    for columna in columnas_categoricas:
        n_unknown = (dataset[columna] == 'unknown').sum()
        porcentaje_unknown = (n_unknown / len(dataset)) * 100
        
        if n_unknown > 0:
            print(f"\nColumna '{columna}': {n_unknown} valores 'unknown' ({porcentaje_unknown:.2f}%)")
            
            if porcentaje_unknown > umbral:
                columnas_con_unknown.append({
                    'columna': columna,
                    'porcentaje': porcentaje_unknown
                })
                print(f"  ALERTA: Excede el umbral del {umbral}%")
    
    if columnas_con_unknown:
        print(f"\n{'='*80}")
        print(f"RESUMEN: {len(columnas_con_unknown)} columna(s) con valores 'unknown' críticos")
        for item in columnas_con_unknown:
            print(f"  - {item['columna']}: {item['porcentaje']:.2f}%")
        print(f"{'='*80}")
        print("Recomendación: Tratar valores 'unknown' como valores nulos")


def test_no_duplicados(dataset):
    """
    Test 6: Verificar que no hay registros duplicados
    """
    n_duplicados = dataset.duplicated().sum()
    n_total = len(dataset)
    
    porcentaje_duplicados = (n_duplicados / n_total) * 100
    
    print(f"\nRegistros duplicados: {n_duplicados} ({porcentaje_duplicados:.2f}%)")
    
    # Esperamos menos del 1% de duplicados
    assert porcentaje_duplicados < 1.0, f"Hay demasiados duplicados: {porcentaje_duplicados:.2f}%"


def test_edad_positiva(dataset):
    """
    Test 7: Verificar que la edad es siempre positiva y está en un rango razonable
    """
    edades_invalidas = ((dataset['age'] < 18) | (dataset['age'] > 100)).sum()
    
    print(f"\nEdades fuera del rango [18-100]: {edades_invalidas}")
    print(f"Edad mínima: {dataset['age'].min()}, Edad máxima: {dataset['age'].max()}")
    
    assert edades_invalidas == 0, f"Hay {edades_invalidas} valores de edad fuera del rango esperado (18-100)"


def test_duracion_positiva(dataset):
    """
    Test 8: Verificar que la duración de las llamadas es positiva
    """
    duraciones_negativas = (dataset['duration'] < 0).sum()
    
    print(f"\nDuraciones negativas: {duraciones_negativas}")
    print(f"Duración mínima: {dataset['duration'].min()}, Duración máxima: {dataset['duration'].max()}")
    
    assert duraciones_negativas == 0, f"Hay {duraciones_negativas} valores de duración negativos"


def test_variable_objetivo_valores_validos(dataset):
    """
    Test 9: Verificar que la variable objetivo 'y' solo tiene valores 'yes' o 'no'
    """
    valores_validos = {'yes', 'no'}
    valores_invalidos = ~dataset['y'].isin(valores_validos)
    n_invalidos = valores_invalidos.sum()
    
    print(f"\nValores únicos en 'y': {dataset['y'].unique()}")
    print(f"Distribución de 'y': {dataset['y'].value_counts().to_dict()}")
    
    assert n_invalidos == 0, f"La variable objetivo 'y' tiene {n_invalidos} valores inesperados"


def test_campaign_minimo(dataset):
    """
    Test 10: Verificar que campaign es al menos 1
    """
    campaign_invalido = (dataset['campaign'] < 1).sum()
    
    print(f"\nValores de campaign < 1: {campaign_invalido}")
    print(f"Campaign mínimo: {dataset['campaign'].min()}, Campaign máximo: {dataset['campaign'].max()}")
    
    assert campaign_invalido == 0, f"Hay {campaign_invalido} valores de campaign menores a 1"


def test_coherencia_pdays_previous(dataset):
    """
    Test 11: Verificar coherencia entre pdays y previous
    Si pdays = 999 (nunca contactado), previous debería ser 0
    """
    # Filtrar registros donde pdays = 999
    filas_nunca_contactado = dataset[dataset['pdays'] == 999]
    
    # En estas filas, previous debería ser 0
    inconsistencias = (filas_nunca_contactado['previous'] != 0).sum()
    
    porcentaje_inconsistencias = (inconsistencias / len(filas_nunca_contactado)) * 100
    
    print(f"\nInconsistencias pdays/previous: {inconsistencias} ({porcentaje_inconsistencias:.2f}%)")
    print(f"Total de registros con pdays=999: {len(filas_nunca_contactado)}")
    
    # Permitimos hasta un 5% de inconsistencias (puede haber razones válidas)
    assert porcentaje_inconsistencias < 5.0, f"Demasiadas inconsistencias: {porcentaje_inconsistencias:.2f}%"


def test_cardinalidad_variables_categoricas(dataset):
    """
    Test 12: Verificar que las variables categóricas tienen variabilidad adecuada
    """
    columnas_categoricas = dataset.select_dtypes(include=['object']).columns
    
    print(f"\n{'='*80}")
    print("CARDINALIDAD DE VARIABLES CATEGÓRICAS")
    print(f"{'='*80}")
    
    baja_variabilidad = []
    
    for columna in columnas_categoricas:
        n_unicos = dataset[columna].nunique()
        n_total = len(dataset)
        
        print(f"\n{columna}: {n_unicos} valores únicos ({(n_unicos/n_total)*100:.2f}%)")
        
        # Las variables categóricas deberían tener al menos 2 valores únicos
        if n_unicos < 2:
            baja_variabilidad.append(columna)
            print(f"  ALERTA: Muy poca variabilidad")
    
    assert len(baja_variabilidad) == 0, f"Columnas con baja variabilidad (< 2 valores): {baja_variabilidad}"


def test_rango_variables_numericas(dataset):
    """
    Test 13: Verificar que las variables numéricas están en rangos razonables
    """
    print(f"\n{'='*80}")
    print("VALIDACIÓN DE RANGOS NUMÉRICOS")
    print(f"{'='*80}")
    
    # emp.var.rate: tasa de variación del empleo (típicamente entre -5 y 5)
    emp_fuera_rango = ((dataset['emp.var.rate'] < -5) | (dataset['emp.var.rate'] > 5)).sum()
    print(f"\nemp.var.rate fuera de rango [-5, 5]: {emp_fuera_rango}")
    assert emp_fuera_rango == 0, f"emp.var.rate tiene {emp_fuera_rango} valores fuera del rango esperado"
    
    # cons.price.idx: índice de precios al consumidor (típicamente entre 90 y 100)
    price_fuera_rango = ((dataset['cons.price.idx'] < 90) | (dataset['cons.price.idx'] > 100)).sum()
    print(f"cons.price.idx fuera de rango [90, 100]: {price_fuera_rango}")
    assert price_fuera_rango == 0, f"cons.price.idx tiene {price_fuera_rango} valores fuera del rango esperado"
    
    # euribor3m: tasa euribor a 3 meses (típicamente entre 0 y 10)
    euribor_fuera_rango = ((dataset['euribor3m'] < 0) | (dataset['euribor3m'] > 10)).sum()
    print(f"euribor3m fuera de rango [0, 10]: {euribor_fuera_rango}")
    assert euribor_fuera_rango == 0, f"euribor3m tiene {euribor_fuera_rango} valores fuera del rango esperado"
    
    print(f"\n{'='*80}")
    print("Todos los rangos numéricos son válidos")
    print(f"{'='*80}")


# Función para mostrar información sobre cómo ejecutar los tests
def test_info():
    """
    Test informativo: Muestra cómo ejecutar los tests
    """
    print("\n" + "=" * 80)
    print("INFORMACIÓN DE EJECUCIÓN")
    print("=" * 80)
    print("\nPara ejecutar estos tests, usa uno de estos comandos:")
    print("  pytest test/test_data_quality.py -v")
    print("  pytest test/test_data_quality.py -v -s  # Con salida detallada")
    print("\nPara generar un reporte HTML:")
    print("  pytest test/test_data_quality.py --html=reporte_calidad.html")
    print("=" * 80 + "\n")
    
    # Este test siempre pasa
    assert True


if __name__ == "__main__":
    print("Ejecuta los tests usando: pytest test/test_data_quality.py -v")