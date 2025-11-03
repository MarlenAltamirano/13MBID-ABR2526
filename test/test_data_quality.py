"""
Test de Calidad de Datos usando pytest y Great Expectations
Sesión Práctica 04 - Metodologías de Gestión y Diseño de Proyectos Big Data

Este script implementa tests automatizados para evaluar la calidad del dataset
usando Great Expectations, que es la herramienta estándar en la industria.
"""

import pytest
import pandas as pd
import great_expectations as gx
from great_expectations.dataset import PandasDataset


# Fixture de pytest para cargar los datos una sola vez
@pytest.fixture(scope="module")
def dataset():
    """
    Carga el dataset y lo convierte en un PandasDataset de Great Expectations
    """
    ruta_archivo = 'data/raw/bank-additional-full.csv'
    df = pd.read_csv(ruta_archivo, sep=';')
    
    # Convertir a PandasDataset para usar Great Expectations
    ge_df = PandasDataset(df)
    
    return ge_df


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
    resultado = dataset.expect_table_columns_to_match_ordered_list(
        column_list=columnas_esperadas
    )
    
    assert resultado.success, f"Las columnas no coinciden con las esperadas"


def test_esquema_dataframe(dataset):
    """
    Test 2: Validar el esquema completo del dataframe (tipos de datos esperados)
    """
    df_pandas = dataset._data_asset
    
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
        if columna not in df_pandas.columns:
            errores.append(f" Columna '{columna}' no existe en el dataset")
            print(f" Columna '{columna}' no existe en el dataset")
        else:
            tipo_actual = str(df_pandas[columna].dtype)
            if tipo_actual == tipo_esperado:
                print(f" '{columna}': {tipo_actual}")
            else:
                errores.append(f"✗ Columna '{columna}': esperado {tipo_esperado}, encontrado {tipo_actual}")
                print(f" '{columna}': esperado {tipo_esperado}, encontrado {tipo_actual}")
    
    print("="*80)
    
    # Verificar si hay columnas adicionales no esperadas
    columnas_extra = set(df_pandas.columns) - set(esquema_esperado.keys())
    if columnas_extra:
        print(f"\n Columnas adicionales no esperadas: {columnas_extra}")
        errores.append(f"Columnas adicionales: {columnas_extra}")
    
    # Si hay errores, el test falla
    assert len(errores) == 0, f"\nErrores encontrados en el esquema:\n" + "\n".join(errores)


def test_no_filas_vacias(dataset):
    """
    Test 3: Verificar que el dataset no está vacío
    """
    resultado = dataset.expect_table_row_count_to_be_between(
        min_value=1000,  # Esperamos al menos 1000 filas
        max_value=50000  # Y menos de 50000
    )
    
    assert resultado.success, f"El dataset no tiene un número razonable de filas"


def test_valores_nulos_todas_columnas(dataset):
    """
    Test 4: Verificar el porcentaje de valores nulos en todas las columnas
    Identificar columnas que excedan el umbral del 20%
    """
    df_pandas = dataset._data_asset
    umbral = 20  # Umbral del 20% según la memoria de trabajo
    
    columnas_problematicas = []
    
    for columna in df_pandas.columns:
        # Verificar que la columna no tiene más del 20% de valores nulos
        resultado = dataset.expect_column_values_to_not_be_null(
            column=columna,
            mostly=0.80  # Esperamos que al menos el 80% NO sean nulos (= máximo 20% nulos)
        )
        
        # Calcular el porcentaje de valores nulos
        n_nulos = df_pandas[columna].isnull().sum()
        porcentaje_nulos = (n_nulos / len(df_pandas)) * 100
        
        if porcentaje_nulos > 0:
            print(f"\nColumna '{columna}': {n_nulos} valores nulos ({porcentaje_nulos:.2f}%)")
        
        # Si excede el umbral, agregar a la lista de columnas problemáticas
        if porcentaje_nulos > umbral:
            columnas_problematicas.append({
                'columna': columna,
                'porcentaje': porcentaje_nulos
            })
            print(f"ALERTA: Excede el umbral del {umbral}%")
    
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
    
    for columna in columnas_categoricas:
        # Verificar que NO todos los valores son 'unknown'
        resultado = dataset.expect_column_values_to_not_match_regex(
            column=columna,
            regex='unknown',
            mostly=0.85  # Esperamos que al menos 85% NO sean 'unknown'
        )
        
        if not resultado.success:
            unexpected_percent = resultado.result.get('unexpected_percent', 0)
            print(f"\nColumna '{columna}' tiene {unexpected_percent:.2f}% de valores 'unknown'")


def test_no_duplicados(dataset):
    """
    Test 6: Verificar que no hay registros duplicados
    """
    # Contar duplicados manualmente porque Great Expectations
    # no tiene una expectativa directa para esto
    df_pandas = dataset._data_asset
    n_duplicados = df_pandas.duplicated().sum()
    n_total = len(df_pandas)
    
    porcentaje_duplicados = (n_duplicados / n_total) * 100
    
    print(f"\nRegistros duplicados: {n_duplicados} ({porcentaje_duplicados:.2f}%)")
    
    # Esperamos menos del 1% de duplicados
    assert porcentaje_duplicados < 1.0, f"Hay demasiados duplicados: {porcentaje_duplicados:.2f}%"


def test_edad_positiva(dataset):
    """
    Test 7: Verificar que la edad es siempre positiva
    """
    resultado = dataset.expect_column_values_to_be_between(
        column='age',
        min_value=18,
        max_value=100
    )
    
    assert resultado.success, "Hay valores de edad fuera del rango esperado (18-100)"


def test_duracion_positiva(dataset):
    """
    Test 8: Verificar que la duración de las llamadas es positiva
    """
    resultado = dataset.expect_column_values_to_be_between(
        column='duration',
        min_value=0,
        max_value=None  # Sin límite superior
    )
    
    assert resultado.success, "Hay valores de duración negativos"


def test_variable_objetivo_valores_validos(dataset):
    """
    Test 9 Verificar que la variable objetivo 'y' solo tiene valores 'yes' o 'no'
    """
    resultado = dataset.expect_column_values_to_be_in_set(
        column='y',
        value_set=['yes', 'no']
    )
    
    assert resultado.success, "La variable objetivo 'y' tiene valores inesperados"


def test_campaign_minimo(dataset):
    """
    Test 10: Verificar que campaign es al menos 1
    """
    resultado = dataset.expect_column_values_to_be_between(
        column='campaign',
        min_value=1,
        max_value=None
    )
    
    assert resultado.success, "Hay valores de campaign menores a 1"


def test_coherencia_pdays_previous(dataset):
    """
    Test 11: Verificar coherencia entre pdays y previous
    Si pdays = 999 (nunca contactado), previous debería ser 0
    """
    df_pandas = dataset._data_asset
    
    # Filtrar registros donde pdays = 999
    filas_nunca_contactado = df_pandas[df_pandas['pdays'] == 999]
    
    # En estas filas, previous debería ser 0
    inconsistencias = (filas_nunca_contactado['previous'] != 0).sum()
    
    porcentaje_inconsistencias = (inconsistencias / len(filas_nunca_contactado)) * 100
    
    print(f"\nInconsistencias pdays/previous: {inconsistencias} ({porcentaje_inconsistencias:.2f}%)")
    
    # Permitimos hasta un 5% de inconsistencias (puede haber razones válidas)
    assert porcentaje_inconsistencias < 5.0, f"Demasiadas inconsistencias: {porcentaje_inconsistencias:.2f}%"


def test_cardinalidad_variables_categoricas(dataset):
    """
    Test 12: Verificar que las variables categóricas tienen variabilidad adecuada
    """
    df_pandas = dataset._data_asset
    columnas_categoricas = df_pandas.select_dtypes(include=['object']).columns
    
    for columna in columnas_categoricas:
        n_unicos = df_pandas[columna].nunique()
        n_total = len(df_pandas)
        
        # Las variables categóricas deberían tener al menos 2 valores únicos
        # (excepto si es la variable objetivo con solo yes/no)
        assert n_unicos >= 2, f"La columna '{columna}' tiene muy poca variabilidad: {n_unicos} valores únicos"
        
        print(f"\nColumna '{columna}': {n_unicos} valores únicos ({(n_unicos/n_total)*100:.2f}%)")


def test_rango_variables_numericas(dataset):
    """
    Test 13: Verificar que las variables numéricas están en rangos razonables
    """
    # emp.var.rate: tasa de variación del empleo (típicamente entre -5 y 5)
    resultado_emp = dataset.expect_column_values_to_be_between(
        column='emp.var.rate',
        min_value=-5,
        max_value=5
    )
    assert resultado_emp.success, "emp.var.rate tiene valores fuera del rango esperado"
    
    # cons.price.idx: índice de precios al consumidor (típicamente entre 90 y 95)
    resultado_price = dataset.expect_column_values_to_be_between(
        column='cons.price.idx',
        min_value=90,
        max_value=100
    )
    assert resultado_price.success, "cons.price.idx tiene valores fuera del rango esperado"
    
    # euribor3m: tasa euribor a 3 meses (típicamente entre 0 y 6)
    resultado_euribor = dataset.expect_column_values_to_be_between(
        column='euribor3m',
        min_value=0,
        max_value=10
    )
    assert resultado_euribor.success, "euribor3m tiene valores fuera del rango esperado"


# Función para ejecutar todos los tests y generar un reporte
def generar_reporte_calidad():
    """
    Genera un reporte HTML con todos los resultados de calidad
    Este método se puede ejecutar aparte de pytest
    """
    print("\n" + "=" * 80)
    print("Para ejecutar estos tests, usa el comando:")
    print("  pytest test_data_quality.py -v")
    print("\nPara generar un reporte HTML:")
    print("  pytest test_data_quality.py --html=reporte_calidad.html")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    generar_reporte_calidad()