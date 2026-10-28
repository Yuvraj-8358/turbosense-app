import json
import os
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
KEEP_TREES = int(os.getenv("TURBOSENSE_KEEP_TREES", "16"))
if KEEP_TREES < 1:
    raise ValueError("TURBOSENSE_KEEP_TREES must be at least 1")


def compact_estimator(estimator, label: str) -> bool:
    estimators = getattr(estimator, "estimators_", None)
    if not estimators or len(estimators) <= KEEP_TREES:
        return False

    if KEEP_TREES == 1:
        selected = [estimators[len(estimators) // 2]]
    else:
        last_index = len(estimators) - 1
        indices = [round(i * last_index / (KEEP_TREES - 1)) for i in range(KEEP_TREES)]
        selected = [estimators[index] for index in indices]
    estimator.estimators_ = selected
    estimator.n_estimators = len(selected)
    print(f"Compacted {label}: {len(estimators)} -> {len(selected)} trees")
    return True


def compact_bundle(path: Path) -> None:
    original_size = path.stat().st_size
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
        print(
            f"Saved {path.name}: {original_size:,} -> {path.stat().st_size:,} bytes"
        )


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for subset in manifest["datasets"]:
        compact_bundle(MODEL_DIR / f"{subset}_model.joblib")


if __name__ == "__main__":
    main()
