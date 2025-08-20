# uvicorn app:app --host 0.0.0.0 --port 8000 --reload

from fastapi import FastAPI, HTTPException, Request
import pandas as pd
import joblib
import os

app = FastAPI(title="BTC LogisticRegression API", version="1.0")

MODEL_PATH = "btc_latest.joblib"
model = None
ORIGINAL_COLUMNS = None
CLEAN_COLUMNS = None
COL_MAP = None


# =======================
# Carga del modelo
# =======================
@app.on_event("startup")
def load_model():
    global model, ORIGINAL_COLUMNS, CLEAN_COLUMNS, COL_MAP
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    if not hasattr(model, "feature_names_in_"):
        raise ValueError("El modelo no tiene información de columnas de entrada.")

    ORIGINAL_COLUMNS = list(model.feature_names_in_)
    CLEAN_COLUMNS = [c.strip() for c in ORIGINAL_COLUMNS]
    COL_MAP = {clean: orig for clean, orig in zip(CLEAN_COLUMNS, ORIGINAL_COLUMNS)}

    print("✅ Modelo cargado.")
    print("Original:", ORIGINAL_COLUMNS)
    print("Limpias :", CLEAN_COLUMNS)


# =======================
# Utilidad: extraer parámetros de query
# =======================
def extract_query_params(request: Request):
    kwargs = dict(request.query_params)

    # extra → symbol y fecha
    symbol = kwargs.pop("symbol", None)
    fecha = kwargs.pop("fecha", None)

    # validar que estén presentes
    if not symbol:
        raise HTTPException(status_code=400, detail="El parámetro 'symbol' es obligatorio")
    if not fecha:
        raise HTTPException(status_code=400, detail="El parámetro 'fecha' es obligatorio")

    # limpiar nombres de features
    cleaned_kwargs = {k.strip(): v for k, v in kwargs.items()}

    # validar columnas faltantes
    missing = [col for col in CLEAN_COLUMNS if col not in cleaned_kwargs]
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")

    # construir fila en orden
    row = []
    for col in CLEAN_COLUMNS:
        val = cleaned_kwargs[col]
        try:
            row.append(float(val.strip()))
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"El parámetro '{col}' no es numérico: '{val}'"
            )

    df = pd.DataFrame([row], columns=CLEAN_COLUMNS)
    df.columns = [COL_MAP[c] for c in CLEAN_COLUMNS]

    return df, symbol, fecha


# =======================
# Endpoint de predicción
# =======================
@app.get("/predict")
def predict(request: Request):
    try:
        df, symbol, fecha = extract_query_params(request)

        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0].tolist()

        return {
            "prediction": int(pred),
            "probabilities": proba
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/features")
def get_features():
    return {"features": CLEAN_COLUMNS}
