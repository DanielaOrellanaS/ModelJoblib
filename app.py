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
CLIENT_TO_MODEL = None


# =======================
# Carga del modelo
# =======================
@app.on_event("startup")
def load_model():
    global model, ORIGINAL_COLUMNS, CLEAN_COLUMNS, CLIENT_TO_MODEL

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # nombres que acepta el cliente (query string)
    CLEAN_COLUMNS = [
        "o5","c5","c5d","h5","l5","v5",
        "o15","c15","h15","l15","v15",
        "r5","r15","m5","s5","m15","s15",
        "ema550","ema5200","ema50_prev","ema5200_prev",
        "macdLine5","signalLine5","macdLine_prev5","signalLine_prev5",
        "adx5","diPlus5","diMinus5",
        "ema5015","ema20015","ema50_prev15","ema200_prev15",
        "macdLine15","signalLine15","macdLine_prev15","signalLine_prev15",
        "adx15","diPlus15","diMinus15"
    ]

    # nombres reales del modelo (usados al entrenar)
    ORIGINAL_COLUMNS = list(model.feature_names_in_)
    # diccionario de mapeo cliente → modelo
    CLIENT_TO_MODEL = {
        "o5": " precioopen5",
        "c5": "precioclose5",
        "c5d": "c5d",
        "h5": "preciohigh5",
        "l5": "preciolow5",
        "v5": "volume5",

        "o15": "precioopen15",
        "c15": "precioclose15",
        "h15": "preciohigh15",
        "l15": "preciolow15",
        "v15": "volume15",

        "r5": "rsi5",
        "r15": "rsi15",
        "m5": "iStochaMain5",
        "s5": "iStochaSign5",
        "m15": "iStochaMain15",
        "s15": "iStochaSign15",

        "iBBs5": "iBBs5",
        "iBBi5": "iBBi5",
        "iBBs15": "iBBs15",
        "iBBi15": "iBBi15",

        "ema550": "ema550",
        "ema5200": "ema5200",
        "ema50_prev": "ema50_prev",
        "ema5200_prev": "ema5200_prev",

        "macdLine5": "macdLine5",
        "signalLine5": "signalLine5",
        "macdLine_prev5": "macdLine_prev5",
        "signalLine_prev5": "signalLine_prev5",

        "adx5": "adx5",
        "diPlus5": "diPlus5",
        "diMinus5": "diMinus5",

        "ema5015": "ema5015",
        "ema20015": "ema20015",
        "ema50_prev15": "ema50_prev15",
        "ema200_prev15": "ema200_prev15",

        "macdLine15": "macdLine15",
        "signalLine15": "signalLine15",
        "macdLine_prev15": "macdLine_prev15",
        "signalLine_prev15": "signalLine_prev15",

        "adx15": "adx15 ",
        "diPlus15": "diPlus15",
        "diMinus15": "diMinus15"
    }

    print("✅ Modelo cargado con mapeo cliente → modelo.")
    print("Original:", ORIGINAL_COLUMNS)
    print("Cliente :", CLEAN_COLUMNS)

# =======================
# Utilidad: extraer parámetros de query
# =======================
def extract_query_params(request: Request):
    kwargs = dict(request.query_params)

    # extra → symbol y fecha
    symbol = kwargs.pop("symbol", None)
    fecha = kwargs.pop("fecha", None)

    if not symbol:
        raise HTTPException(status_code=400, detail="El parámetro 'symbol' es obligatorio")
    if not fecha:
        raise HTTPException(status_code=400, detail="El parámetro 'fecha' es obligatorio")

    # limpiar nombres de features
    cleaned_kwargs = {k.strip(): v for k, v in kwargs.items()}

    # validar columnas faltantes (cliente)
    missing = [col for col in CLEAN_COLUMNS if col not in cleaned_kwargs]
    if missing:
        raise HTTPException(status_code=400, detail=f"Faltan columnas: {missing}")

    # construir fila en orden del modelo
    row = []
    for model_col in ORIGINAL_COLUMNS:
        # buscar la clave cliente que corresponde a esta columna del modelo
        client_col = next((k for k, v in CLIENT_TO_MODEL.items() if v == model_col), None)
        if client_col is None or client_col not in cleaned_kwargs:
            raise HTTPException(status_code=400, detail=f"Falta el parámetro para '{model_col}'")

        val = cleaned_kwargs[client_col]
        try:
            row.append(float(val.strip()))
        except Exception:
            raise HTTPException(status_code=400, detail=f"El parámetro '{client_col}' no es numérico: '{val}'")

    df = pd.DataFrame([row], columns=ORIGINAL_COLUMNS)
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
    return {"features_cliente": CLEAN_COLUMNS, "features_modelo": ORIGINAL_COLUMNS}
