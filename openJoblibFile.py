# open_joblib_info.py
import joblib
import os

# Ruta al archivo joblib
ruta_modelo = "btc_latest.joblib"  # <-- cámbialo por el nombre real

if not os.path.exists(ruta_modelo):
    raise FileNotFoundError(f"No se encontró el archivo: {ruta_modelo}")

# Cargar modelo
modelo = joblib.load(ruta_modelo)

print("="*60)
print(f"Tipo de objeto cargado: {type(modelo)}")
print("="*60)

# Mostrar parámetros si existen
if hasattr(modelo, "get_params"):
    print("\n📌 Parámetros del modelo:")
    for k, v in modelo.get_params().items():
        print(f"  {k}: {v}")

# Mostrar clases si es un clasificador
if hasattr(modelo, "classes_"):
    print("\n📌 Clases que predice:", modelo.classes_)

# Mostrar coeficientes si es un modelo lineal
if hasattr(modelo, "coef_"):
    print("\n📌 Forma de los coeficientes:", modelo.coef_.shape)
    print("Primeros coeficientes:\n", modelo.coef_[:5])

# Intercepto
if hasattr(modelo, "intercept_"):
    print("\n📌 Interceptos:", modelo.intercept_)

# Si es un Pipeline, mostrar los pasos
if hasattr(modelo, "steps"):
    print("\n📌 Pasos del Pipeline:")
    for paso in modelo.steps:
        print("  -", paso)

# Buscar variables usadas para entrenar
print("\n📌 Intentando identificar variables de entrenamiento...")
if hasattr(modelo, "feature_names_in_"):
    print("Variables usadas:", modelo.feature_names_in_)
else:
    print("No se encontraron nombres de variables directamente en el modelo.")

print("\n✅ Análisis completado.")
