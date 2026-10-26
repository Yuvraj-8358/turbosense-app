import json
import os
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
REPO = os.environ.get("GITHUB_REPOSITORY", "").strip()
TAG = os.environ.get("MODEL_RELEASE_TAG", "").strip()
TOKEN = os.environ.get("GITHUB_TOKEN", "").strip()
if not REPO or not TAG:
    raise SystemExit("Set GITHUB_REPOSITORY (owner/repository) and MODEL_RELEASE_TAG (for example v1.0.0).")
headers = {"Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
if TOKEN:
    headers["Authorization"] = f"Bearer {TOKEN}"
api = f"https://api.github.com/repos/{REPO}/releases/tags/{TAG}"
response = requests.get(api, headers=headers, timeout=30)
response.raise_for_status()
release = response.json()
assets = {asset["name"]: asset["url"] for asset in release.get("assets", [])}
manifest = json.loads((ROOT / "manifest.json").read_text(encoding="utf-8"))
model_dir = ROOT / "models"
model_dir.mkdir(exist_ok=True)
for subset in manifest["datasets"]:
    name = f"{subset}_model.joblib"
    if name not in assets:
        raise SystemExit(f"Release {TAG} is missing required asset {name}")
    asset_headers = dict(headers)
    asset_headers["Accept"] = "application/octet-stream"
    with requests.get(assets[name], headers=asset_headers, stream=True, timeout=(30, 300)) as download:
        download.raise_for_status()
        target = model_dir / name
        temp = target.with_suffix(target.suffix + ".part")
        with temp.open("wb") as output:
            for chunk in download.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
        if temp.stat().st_size == 0:
            temp.unlink(missing_ok=True)
            raise SystemExit(f"Downloaded an empty artifact: {name}")
        temp.replace(target)
    print(f"Downloaded {name} ({target.stat().st_size:,} bytes)")
print("All four model bundles are ready.")
