# TURBOSENSE — C-MAPSS engine health app

TURBOSENSE estimates remaining useful life (RUL) and classifies engine health using the combined regression and classification model bundle for FD001–FD004. The web interface supports manual entry, CSV uploads, and Word .docx uploads.

## Entering sensor readings

Choose the dataset first. The Manual sensor entry form changes to show every operating setting and sensor required by that dataset, using the updated names in manifest.json. Enter a finite number in every field.

CSV uploads must have one header row with the exact feature names for the selected dataset. Each data row represents one engine cycle. Uploads are limited to 5 MB and 200 rows.

Word uploads must be modern Word .docx files. Put the exact sensor names in the first row of a table and one reading per row below it. A single reading can also be written in paragraphs as sensor_name: value or sensor_name = value. Legacy .doc files are not supported.

For a time series, include both unit and cycle columns. TURBOSENSE will predict the latest cycle for each unit. The column names and capitalization must match the manifest.

## API

- GET / — web interface
- GET /healthz — checks that all four model files are present
- GET /api/manifest — sensor names and dataset-specific model features
- POST /api/parse-upload — accepts a multipart CSV or .docx upload with subset and file
- POST /api/predict — accepts JSON: {"subset":"FD001","rows":[{"op_setting_1":0.1}]}; include every model feature required for the selected subset
- GET /docs — interactive API documentation

The prediction response contains estimated RUL cycles and a critical, warning, or healthy health label. Outputs are estimates from NASA C-MAPSS data, not certified aircraft safety or maintenance instructions.

## Render configuration

The checked-in render.yaml sets the Python version, Free plan, build command, start command, and /healthz health check. The build installs dependencies, downloads the model bundles from the configured GitHub Release, and compacts tree-ensemble bundles to 16 evenly sampled trees. Render reported that the previous 64-tree bundles exceeded its 512 MB free-instance memory limit during inference. The smaller forests are a memory-saving deployment compromise and should be rechecked against held-out evaluation metrics before the results are presented as model-performance evidence.

Configure these environment variables in Render:

- GITHUB_REPOSITORY: repository containing the model Release assets
- MODEL_RELEASE_TAG: release tag, such as v1.0.0
- GITHUB_TOKEN: read-only token for a private repository; keep it only in Render and never commit it
- PYTHON_VERSION: set in render.yaml

Keep the large .joblib bundles out of normal Git history. Attach the four model bundles to a GitHub Release instead. The free Render instance has 512 MB RAM and may spin down after inactivity. The browser also handles empty or non-JSON server responses without exposing a raw JSON parsing exception.

## Local run

1. Install packages from requirements.txt.
2. Set GITHUB_REPOSITORY, MODEL_RELEASE_TAG, and, for a private release, GITHUB_TOKEN.
3. Run python download_models.py.
4. Start the app with uvicorn app:app --host 0.0.0.0 --port 8000
