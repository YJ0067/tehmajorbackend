import os
import gc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

# =========================================================
# APP
# =========================================================
app = FastAPI(
    title="Heatwave Forecast API",
    description="7-day India heatwave and temperature forecasting API",
    version="1.0.0"
)

# =========================================================
# CONFIG
# =========================================================
SEQ_LEN = 45
FORECAST_DAYS = 7
GRID_PROB_THRESH = 0.50

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_PATH, "heatwave_full_model.keras")
META_PATH = os.path.join(BASE_PATH, "variables", "grid_region_terrain_mapping.joblib")
DATA_PATH = os.path.join(BASE_PATH, "variables2", "precomputed_arrays.npz")
TIME_IDX_PATH = os.path.join(BASE_PATH, "variables2", "time_idx.joblib")
CLIM_PATH = os.path.join(BASE_PATH, "variables", "climatology.joblib")

print("BASE_PATH:", BASE_PATH)
print("MODEL_PATH exists:", os.path.exists(MODEL_PATH))
print("META_PATH exists:", os.path.exists(META_PATH))
print("DATA_PATH exists:", os.path.exists(DATA_PATH))
print("TIME_IDX_PATH exists:", os.path.exists(TIME_IDX_PATH))
print("CLIM_PATH exists:", os.path.exists(CLIM_PATH))

# =========================================================
# CUSTOM LAYER
# =========================================================
class TransposeBCHW(tf.keras.layers.Layer):
    def call(self, x):
        return tf.transpose(x, [0, 3, 1, 2])

    def get_config(self):
        return {}

# =========================================================
# HELPERS
# =========================================================
def load_assets():
    meta = joblib.load(META_PATH)

    lats = meta["lats"]
    lons = meta["lons"]
    india_mask = meta["india_mask"]
    region_id_grid = meta["region_id_grid"]
    regions = meta["regions"]

    H, W = india_mask.shape

    model = load_model(
        MODEL_PATH,
        custom_objects={"TransposeBCHW": TransposeBCHW},
        compile=False
    )

    with np.load(DATA_PATH, allow_pickle=False) as data:
        dyn_scaled = data["dyn_scaled"]
        date_maps = data["date_maps"]
        STATIC_SEQ = data["STATIC_SEQ"]

    time_idx = pd.to_datetime(joblib.load(TIME_IDX_PATH))
    climatology = joblib.load(CLIM_PATH)

    return {
        "lats": lats,
        "lons": lons,
        "india_mask": india_mask,
        "region_id_grid": region_id_grid,
        "regions": regions,
        "H": H,
        "W": W,
        "model": model,
        "dyn_scaled": dyn_scaled,
        "date_maps": date_maps,
        "STATIC_SEQ": STATIC_SEQ,
        "time_idx": time_idx,
        "climatology": climatology
    }

# =========================================================
# ROUTES
# =========================================================
@app.get("/")
def home():
    return {
        "status": "success",
        "message": "Heatwave Forecast API is running"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "message": "API is running",
        "forecast_days": FORECAST_DAYS
    }

@app.get("/predict")
def predict():
    try:
        import requests

        print("Loading assets inside /predict...")

        print("Loading meta...")
        meta = joblib.load(META_PATH)

        lats = meta["lats"]
        lons = meta["lons"]
        india_mask = meta["india_mask"]
        region_id_grid = meta["region_id_grid"]
        regions = meta["regions"]

        H, W = india_mask.shape

        print("Meta loaded successfully")
        print("Grid shape:", H, "x", W)

        print("Loading model...")
        model = load_model(
            MODEL_PATH,
            custom_objects={"TransposeBCHW": TransposeBCHW},
            compile=False
        )

        print("Model loaded successfully")
        print("Model input shape:", model.input_shape)

        print("Loading NPZ file...")
        data = np.load(DATA_PATH, mmap_mode="r")

        print("Creating references to dyn_scaled and date_maps...")
        dyn_scaled_all = data["dyn_scaled"]
        date_maps_all = data["date_maps"]
        STATIC_SEQ = data["STATIC_SEQ"]

        print("Loading time_idx...")
        time_idx = pd.to_datetime(joblib.load(TIME_IDX_PATH))

        print("Loading climatology...")
        climatology = joblib.load(CLIM_PATH)

        print("All assets loaded successfully")

        # =========================================================
        # USE TODAY'S DATE
        # =========================================================
        today_str = datetime.now().strftime("%Y-%m-%d")
        TARGET_DATE = pd.Timestamp(today_str)

        print("Forecast date:", TARGET_DATE.date())

        # =========================================================
        # FIND DATE INDEX
        # =========================================================
        if TARGET_DATE not in time_idx.values:
            idx = int(np.argmin(np.abs(time_idx - TARGET_DATE)))
            print("Exact date not found. Using closest available date.")
        else:
            idx = int(np.where(time_idx == TARGET_DATE)[0][0])

        print("Using index:", idx)
        print("Matched date:", time_idx[idx])

        if idx < SEQ_LEN:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough history before {TARGET_DATE.date()}"
            )

        # =========================================================
        # PREPARE INPUT
        # =========================================================
        hist = slice(idx - SEQ_LEN + 1, idx + 1)

        print("Loading only required history slice...")
        X_dyn = dyn_scaled_all[hist]
        X_date = date_maps_all[hist]

        print("X_dyn shape:", X_dyn.shape)
        print("X_date shape:", X_date.shape)
        print("STATIC_SEQ shape:", STATIC_SEQ.shape)

        print("Concatenating features...")
        X = np.concatenate(
            [X_dyn, X_date, STATIC_SEQ],
            axis=-1
        ).astype(np.float32)

        print("Combined X shape:", X.shape)

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        X *= india_mask[None, :, :, None].astype(np.float32)

        X_input = X[None, ...]

        print("Model input shape after batch dimension:", X_input.shape)

        # =========================================================
        # MODEL PREDICTION
        # =========================================================
        print("Running model prediction...")
        pr_reg, pr_cls = model.predict(X_input, verbose=0)

        print("Prediction complete")

        pr_reg = pr_reg[0, :, :, :, 0]
        pr_cls = pr_cls[0, :, :, :, 0]

        print("pr_reg shape:", pr_reg.shape)
        print("pr_cls shape:", pr_cls.shape)

        # =========================================================
        # HYBRID FORECAST
        # =========================================================
        print("Generating hybrid predictions...")
        hybrid_preds = []

        for day in range(FORECAST_DAYS):
            month = (TARGET_DATE + timedelta(days=day)).month
            base = climatology["clim_tmax"][month - 1]
            correction = pr_reg[day] * 0.3
            final = (base + correction) * india_mask
            hybrid_preds.append(final)

        hybrid_preds = np.array(hybrid_preds)

        print("Hybrid predictions shape:", hybrid_preds.shape)

        # =========================================================
        # CREATE RESPONSE DATA
        # =========================================================
        print("Creating response rows...")
        rows = []

        for L in range(1, FORECAST_DAYS + 1):
            target_day = TARGET_DATE + timedelta(days=L - 1)

            for h in range(H):
                for w in range(W):
                    if india_mask[h, w] == 0:
                        continue

                    lat = float(lats[h])
                    lon = float(lons[w])

                    t_pred = float(hybrid_preds[L - 1, h, w])
                    hw_prob = float(pr_cls[L - 1, h, w])
                    hw_pred = int(hw_prob >= GRID_PROB_THRESH)

                    region_id = int(region_id_grid[h, w])
                    region_name = regions[region_id]

                    rows.append({
                        "issue_date": str(TARGET_DATE.date()),
                        "date": str(target_day.date()),
                        "lead": L,
                        "lat": lat,
                        "lon": lon,
                        "region_id": region_id,
                        "region_name": region_name,
                        "tmax_pred": round(t_pred, 2),
                        "hw_pred": hw_pred,
                        "hw_prob": round(hw_prob, 4)
                    })

        print("Total rows created:", len(rows))

        payload = {
            "status": "success",
            "generated_at": str(TARGET_DATE.date()),
            "total_records": len(rows),
            "data": rows
        }

        # =========================================================
        # POST TO SHAREBRO
        # =========================================================
        API_URL = "https://sharebro.onrender.com/api/forecast"

        print("Sending payload to Sharebro...")
        sharebro_response = requests.post(
            API_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=60
        )

        print("Sharebro response code:", sharebro_response.status_code)

        response = {
            "status": "success",
            "sharebro_status_code": sharebro_response.status_code,
            "sharebro_response": sharebro_response.text,
            "payload": payload
        }

        print("Cleaning memory...")
        del model, data, dyn_scaled_all, date_maps_all, STATIC_SEQ
        del X_dyn, X_date, X, X_input, pr_reg, pr_cls, hybrid_preds
        del rows, payload, meta, time_idx, climatology
        gc.collect()

        return response

    except HTTPException:
        raise
    except Exception as e:
        print("ERROR:", str(e))
        gc.collect()
        raise HTTPException(status_code=500, detail=str(e))