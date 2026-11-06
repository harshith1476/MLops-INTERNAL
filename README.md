# E-Commerce Product Return Prediction System (CI/CD)

Predicts whether an order will be **Returned** or **Not Returned**, served with FastAPI, tested and
built automatically with GitHub Actions, and published as a Docker image on GitHub Container Registry.

## Structure
```
data/ecommerce_returns.csv     dataset (1200 rows)
src/features.py                cleaning + feature engineering (shared by training and API)
src/train.py                   trains LogisticRegression + RandomForest, saves best model
app/main.py                    FastAPI app: GET /health, POST /predict
models/return_model.joblib     trained pipeline (preprocessing + model)
models/metrics.json            evaluation results
tests/                         pytest tests
Dockerfile, .dockerignore      container build
.github/workflows/ci-cd.yml    CI/CD pipeline
```

## Run locally
```
pip install -r requirements-dev.txt
python -m src.train
pytest -v
uvicorn app.main:app --reload      # http://127.0.0.1:8000/docs
```

## Run with Docker
```
docker run -p 8000:8000 ghcr.io/<github-username>/<repo-name>:latest
```
