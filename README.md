# TURBOSENSE — Render app starter

This FastAPI app uses the four combined model bundles from the supplied Colab artifact ZIP. Each `FD00x_model.joblib` contains both the RUL regressor and health classifier. The matching manifest and sensor symbols are included here. The app caches only one subset bundle at a time to limit memory use.

## Important model-file handling

The supplied `FD004_model.joblib` is about 156 MB, above GitHub's 100 MB ordinary file limit. Keep the model bundles out of the app source repository. Upload the four `.joblib` files as assets to a GitHub Release in a model repository. GitHub documents the 100 MB normal file limit and recommends Git LFS or external storage for large binaries; using Release assets here avoids placing binaries in Git history. [GitHub repository limits](https://docs.github.com/en/repositories/creating-and-managing-repositories/repository-limits)

The app downloads release assets during Render's build. For a private model repository, create a read-only GitHub token and enter it only in Render's environment settings. Never commit the token.

## Prepare the GitHub Release

1. Create a model repository on GitHub.
2. Create a release tagged `v1.0.0`.
3. From the ZIP's `deployment_artifacts/` directory, attach exactly `FD001_model.joblib`, `FD002_model.joblib`, `FD003_model.joblib`, and `FD004_model.joblib` as release assets.
4. Keep the release and app manifest from the same Colab run.

## Deploy the app

1. Create a separate GitHub repository for the contents of this folder. Do not add the `models/` folder or any `.joblib` files to that code repo.
2. In Render, create a Web Service connected to the app repository. The included `render.yaml` specifies the Python version, build/start commands, and health route. Render builds web services from connected Git repositories and the app binds to `0.0.0.0:$PORT`. [Render web services](https://render.com/docs/web-services)
3. Set these Render environment variables:
   - `GITHUB_REPOSITORY`: `owner/model-repository`
   - `MODEL_RELEASE_TAG`: `v1.0.0`
   - `GITHUB_TOKEN`: read-only token for a private model repo; leave unset for a public release
4. Start with the configured free plan. The FD004 bundle is about 156 MB on disk plus Python and model runtime memory; if Render reports an out-of-memory failure, select a higher-memory plan.
5. When build completes, check `/healthz`, open `/`, and submit an engine reading at `/api/predict`. API docs are at `/docs`.

## Local setup

Install `requirements.txt`, set the three variables above, run `python download_models.py`, then start with `uvicorn app:app --host 0.0.0.0 --port 8000`.

## Prediction request

`POST /api/predict` accepts `{ "subset": "FD001", "rows": [{...}] }`. Rows must include every feature required by that subset model, using the documented sensor symbols in `manifest.json`. For a time series, include `unit` and `cycle`; the API predicts the last supplied cycle for each unit. For a single latest-cycle row, those two fields are optional. The app returns RUL cycles and a `critical`, `warning`, or `healthy` label.

## Limitations

The app returns model estimates, not certified aircraft safety or maintenance decisions. Verify artifact and library versions together; this project pins NumPy, pandas, and scikit-learn to the Colab manifest versions.
