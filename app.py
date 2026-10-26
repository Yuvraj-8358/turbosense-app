import json
import math
import os
from typing import Any
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
MANIFEST = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
DATASETS = set(MANIFEST["datasets"])

app = FastAPI(title="TURBOSENSE", version=MANIFEST.get("version", "1.0.0"), description="NASA C-MAPSS RUL and health-state inference")

class PredictionRequest(BaseModel):
    subset: str = Field(description="One of FD001, FD002, FD003, FD004")
    rows: list[dict[str, Any]] = Field(min_length=1, max_length=200, description="One or more engine-cycle rows using the documented feature names")

@lru_cache(maxsize=1)
def load_bundle(subset: str):
    path = MODEL_DIR / f"{subset}_model.joblib"
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact not found: {path.name}")
    bundle = joblib.load(path)
    required = {"regressor", "classifier", "features", "rul_cap"}
    if not required.issubset(bundle):
        raise RuntimeError(f"{path.name} is not a combined TURBOSENSE artifact; missing keys: {sorted(required - set(bundle))}")
    return bundle

@app.get("/", include_in_schema=False)
def home():
    return FileResponse(ROOT / "static" / "index.html")

@app.get("/healthz")
def healthz():
    missing = [f"{ds}_model.joblib" for ds in sorted(DATASETS) if not (MODEL_DIR / f"{ds}_model.joblib").is_file()]
    return {"status": "ok" if not missing else "models_missing", "datasets": sorted(DATASETS), "missing_artifacts": missing}

@app.post("/api/predict")
def predict(request: PredictionRequest):
    subset = request.subset.strip().upper()
    if subset not in DATASETS:
        raise HTTPException(status_code=422, detail="subset must be FD001, FD002, FD003, or FD004")
    try:
        bundle = load_bundle(subset)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load {subset} model bundle: {exc}") from exc
    frame = pd.DataFrame(request.rows)
    missing = [feature for feature in bundle["features"] if feature not in frame.columns]
    if missing:
        raise HTTPException(status_code=422, detail={"message": "Input is missing model features", "missing_features": missing})
    x = frame[bundle["features"]].apply(pd.to_numeric, errors="coerce")
    if x.isna().any().any() or not np.isfinite(x.to_numpy(dtype=float)).all():
        raise HTTPException(status_code=422, detail="Feature values must be finite numbers")
    if "unit" in frame.columns and "cycle" in frame.columns:
        cycles = pd.to_numeric(frame["cycle"], errors="coerce")
        if cycles.isna().any() or not np.isfinite(cycles.to_numpy(dtype=float)).all():
            raise HTTPException(status_code=422, detail="cycle values must be finite numbers when unit and cycle are provided")
        frame["cycle"] = cycles
        frame = frame.sort_values("cycle").groupby("unit", sort=False).tail(1)
        x = frame[bundle["features"]].apply(pd.to_numeric, errors="coerce")
    try:
        rul = np.clip(bundle["regressor"].predict(x), 0, bundle["rul_cap"])
        labels = bundle["classifier"].predict(x)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc
    predictions = [{"predicted_rul": round(float(r), 2), "health": str(label)} for r, label in zip(rul, labels)]
    return {"subset": subset, "prediction_count": len(predictions), "predictions": predictions}
