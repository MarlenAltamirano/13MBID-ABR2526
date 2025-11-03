"""
Test de Calidad de Datos usando Great Expectations

Este script usa un enfoque manual para validar expectativas de calidad de datos
"""

import pandas as pd


def test_great_expectations():
    """Test para verificar que los datos cumplen con las expectativas definidas.
    
    Raises:
        AssertionError: Si alguna de las expectativas no se cumple.
    """
    # Cargar los datos
    df = pd.read_csv("data/raw/bank-additional-full.csv", sep=';')
    
    results = {
        "success": True,
        "expectations": [],
        "statistics": {"success_count": 0, "total_count": 0}
    }
    
    def add_expectation(expectation_name, condition, message=""):
        """Agrega una expectativa al reporte de resultados"""
        results["statistics"]["total_count"] += 1
        if condition:
            results["statistics"]["success_count"] += 1
            results["expectations"].append({
                "expectation": expectation_name,
                "success": True
            })
            print(f"{expectation_name}: PASS")
        else:
            results["success"] = False
            results["expectations"].append({
                "expectation": expectation_name,
                "success": False,
                "message": message
            })
            print(f"{expectation_name}: FAIL - {message}")
    
    print("\n" + "="*80)
    print("VALIDACIONES DE CALIDAD DE DATOS")
    print("="*80 + "\n")
    
    # Validaciones a verificar sobre los datos
    
    # 1. Rango de edad
    add_expectation(
        "age_range",
        df["age"].between(17, 100).all(),
        "La columna 'age' no está en el rango esperado (17-100)."
    )
    
    # 2. Valores válidos en variable objetivo
    add_expectation(
        "target_values",
        df["y"].isin(["yes", "no"]).all(),
        "La columna 'y' contiene valores no válidos."
    )
    
    # 3. Duración no negativa
    add_expectation(
        "duration_positive",
        (df["duration"] >= 0).all(),
        "La columna 'duration' contiene valores negativos."
    )
    
    # 4. Campaign debe ser al menos 1
    add_expectation(
        "campaign_minimum",
        (df["campaign"] >= 1).all(),
        "La columna 'campaign' contiene valores menores a 1."
    )
    
    # 5. Valores nulos en columna 'default' no deben exceder el 20%
    porcentaje_nulos_default = (df["default"].isnull().sum() / len(df)) * 100
    add_expectation(
        "default_null_threshold",
        porcentaje_nulos_default <= 20,
        f"La columna 'default' tiene {porcentaje_nulos_default:.2f}% de valores nulos (límite: 20%)."
    )
    
    # 6. No debe haber registros completamente duplicados
    porcentaje_duplicados = (df.duplicated().sum() / len(df)) * 100
    add_expectation(
        "no_duplicates",
        porcentaje_duplicados < 1.0,
        f"Se encontraron {df.duplicated().sum()} registros duplicados ({porcentaje_duplicados:.2f}%)."
    )
    
    # 7. Variables numéricas económicas en rangos razonables
    add_expectation(
        "emp_var_rate_range",
        df["emp.var.rate"].between(-5, 5).all(),
        "La columna 'emp.var.rate' tiene valores fuera del rango esperado (-5 a 5)."
    )
    
    # 8. Índice de precios al consumidor en rango razonable
    add_expectation(
        "cons_price_idx_range",
        df["cons.price.idx"].between(90, 100).all(),
        "La columna 'cons.price.idx' tiene valores fuera del rango esperado (90-100)."
    )
    
    # 9. Verificar que no hay valores 'unknown' excesivos en 'job'
    porcentaje_unknown_job = ((df["job"] == "unknown").sum() / len(df)) * 100
    add_expectation(
        "job_unknown_threshold",
        porcentaje_unknown_job <= 15,
        f"La columna 'job' tiene {porcentaje_unknown_job:.2f}% de valores 'unknown' (límite: 15%)."
    )
    
    # 10. El dataset debe tener un número razonable de registros
    add_expectation(
        "row_count_reasonable",
        1000 <= len(df) <= 50000,
        f"El dataset tiene {len(df)} registros, fuera del rango esperado (1000-50000)."
    )
    
    # Mostrar resumen
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS")
    print("="*80)
    print(f"Total de expectativas: {results['statistics']['total_count']}")
    print(f"Exitosas: {results['statistics']['success_count']}")
    print(f"Fallidas: {results['statistics']['total_count'] - results['statistics']['success_count']}")
    
    tasa_exito = (results['statistics']['success_count'] / results['statistics']['total_count']) * 100
    print(f"Tasa de éxito: {tasa_exito:.2f}%")
    print("="*80 + "\n")
    
    # Mostrar expectativas fallidas en detalle
    expectativas_fallidas = [exp for exp in results["expectations"] if not exp["success"]]
    if expectativas_fallidas:
        print("EXPECTATIVAS FALLIDAS:")
        for exp in expectativas_fallidas:
            print(f"  • {exp['expectation']}: {exp['message']}")
        print()
    
    # Assertion para que pytest falle si hay expectativas que no se cumplen
    assert results["success"], f"Algunas expectativas de calidad no se cumplieron. Revisa el reporte arriba."


if __name__ == "__main__":
    # Si ejecutas este archivo directamente, ejecuta el test
    test_great_expectations()
    print("Todas las validaciones pasaron exitosamente\n")