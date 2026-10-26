import joblib, numpy as np, pandas as pd
from pathlib import Path
class TurboSensePredictor:
    def __init__(self, artifact_dir): self.artifact_dir=Path(artifact_dir)
    def predict_latest(self, subset, rows):
        subset=subset.upper()
        if subset not in {f"FD{i:03d}" for i in range(1,5)}: raise ValueError("subset must be FD001 through FD004")
        pkg=joblib.load(self.artifact_dir/f"{subset}_model.joblib")
        df=pd.DataFrame(rows)
        missing=[c for c in pkg["features"] if c not in df.columns]
        if missing: raise ValueError(f"Missing model inputs: {missing}")
        if "unit" in df and "cycle" in df: df=df.sort_values("cycle").groupby("unit",sort=False).tail(1)
        x=df[pkg["features"]]
        rul=np.clip(pkg["regressor"].predict(x),0,pkg["rul_cap"])
        health=pkg["classifier"].predict(x)
        return [{"predicted_rul":float(round(r,2)),"health":str(h)} for r,h in zip(rul,health)]