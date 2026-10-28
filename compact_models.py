import json
import os
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parent
MODEL_DIR = Path(os.getenv("MODEL_DIR", ROOT / "models"))
KEEP_TREES = int(os.getenv("TURBOSENSE_KEEP_TREES", "16"))
if KEEP_TREES < 1:
    raise ValueError("TURBOSENSE_KEEP_TREES must be at least 1")


def compact_estimator(estimator, label: str) -> tuple[bool, bool]:
    estimators = getattr(estimator, "estimators_", None)
    if estimators is None or len(estimators) == 0:
        return False, False
    if len(estimators) <= KEEP_TREES:
        print(f"Verified {label}: {len(estimators)} trees (limit {KEEP_TREES})")
        return True, False

    if KEEP_TREES == 1:
        selected = [estimators[len(estimators) // 2]]
    else:
        last_index = len(estimators) - 1
        indices = [round(i * last_index / (KEEP_TREES - 1)) for i in range(KEEP_TREES)]
        selected = [estimators[index] for index in indices]
    estimator.estimators_ = selected
    estimator.n_estimators = len(selected)
    print(f"Compacted {label}: {len(estimators)} -> {len(selected)} trees")
    return True, True


def tree_estimators(root):
    """Find tree ensembles directly or inside a scikit-learn Pipeline."""
    pending = [root]
    seen = set()
    while pending:
        estimator = pending.pop()
        if estimator is None or id(estimator) in seen:
            continue
        seen.add(id(estimator))
        yield estimator
        named_steps = getattr(estimator, "named_steps", None)
        if named_steps:
            pending.extend(named_steps.values())


def compact_bundle(path: Path, require_forest: bool) -> None:
    original_size = path.stat().st_size
    bundle = joblib.load(path)
    changed = False
    found = {}
    for key in ("regressor", "classifier"):
        found[key] = False
        for model in tree_estimators(bundle.get(key)):
            detected, model_changed = compact_estimator(model, f"{path.name}:{key}:{type(model).__name__}")
            found[key] = found[key] or detected
            changed = model_changed or changed
    if require_forest and not all(found.get(key, False) for key in ("regressor", "classifier")):
        raise RuntimeError(
            f"Could not find both tree ensembles in {path.name}; "
            f"regressor={found.get('regressor')}, classifier={found.get('classifier')}"
        )
    if changed:
        temp = path.with_suffix(path.suffix + ".compact")
        joblib.dump(bundle, temp, compress=3)
        temp.replace(path)
        print(
            f"Saved {path.name}: {original_size:,} -> {path.stat().st_size:,} bytes"
        )
    elif require_forest:
        print(f"No compaction needed for {path.name}; both ensembles are within the limit")


def main() -> None:
    manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
    for subset in manifest["datasets"]:
        selected_model = manifest.get("selected_models", {}).get(subset, {}).get("model", "")
        require_forest = selected_model.lower() in {"extratrees", "randomforest"}
        compact_bundle(MODEL_DIR / f"{subset}_model.joblib", require_forest)


if __name__ == "__main__":
    main()
