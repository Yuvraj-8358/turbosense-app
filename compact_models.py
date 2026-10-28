import json
import os
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
KEEP_TREES = int(os.getenv("TURBOSENSE_KEEP_TREES", "64"))


def compact_estimator(estimator, label: str) -> bool:
    estimators = getattr(estimator, "estimators_", None)
    if not estimators or len(estimators) <= KEEP_TREES:
        return False

    step = max(1, len(estimators) // KEEP_TREES)
    selected = list(estimators[::step])[:KEEP_TREES]
    estimator.estimators_ = selected
    estimator.n_estimators = len(selected)
    print(f"Compacted {label}: {len(estimators)} -> {len(selected)} trees")
    return True


def compact_bundle(path: Path) -> None:
    bundle = joblib.load(path)
    changed = False
    for key in ("regressor", "classifier"):
        pipeline = bundle.get(key)
        model = getattr(pipeline, "named_steps", {}).get("model")
        if model is not None:
            changed = compact_estimator(model, f"{path.name}:{key}") or changed
    if changed:
        temp = path.with_suffix(path.suffix + ".compact")
        joblib.dump(bundle, temp, compress=3)
        temp.replace(path)


manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
for subset in manifest["datasets"]:
    compact_bundle(MODEL_DIR / f"{subset}_model.joblib")
