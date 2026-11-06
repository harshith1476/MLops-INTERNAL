# Product Return Prediction using CI/CD – Step-by-Step Guide

Replace `<USER>` with your GitHub username (lowercase) and use repo name `ecom-return-prediction`.
Commands are for **Windows PowerShell** (Docker Desktop must be running). Python 3.11+ required (3.13 recommended).

---------------------------------------------------------------------

## 0. One-time setup

```powershell
# unzip the project, then:
cd ecom-return-prediction
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
```
If activation is blocked: `Set-ExecutionPolicy -Scope Process Bypass`

---------------------------------------------------------------------

## Task 1 – Load dataset and preprocess

**What it does** (`src/features.py → clean()`, `src/train.py`)
- Loads `data/ecommerce_returns.csv` (1200 rows, 14 columns).
- Missing values (36 each) in Price, Discount, Actual_Delivery_Days, Rating, Payment_Method → median (numeric) / most-frequent (categorical) imputation.
- Invalid values → set to NaN, then imputed: Price ≤ 0 (3 rows, value −200), Discount > 100 (2 rows, value 125), rating outside 1–5, negative counts.
- Categorical encoding: One-Hot (Category, Payment, Damaged flag, Package size, Promotion). Target: Returned = 1.

**Command**
```powershell
python -m src.train
```
📸 **SCREENSHOT 1** – Terminal showing `Loaded 1200 rows, 14 columns` and the *Missing values before cleaning* table
(top of the output). Optionally show `src/features.py` `clean()` function in VS Code beside it.

---------------------------------------------------------------------

## Task 2 – Feature engineering

**Features created** (`src/features.py → engineer()`)

| Feature | Formula |
|---|---|
| Discount_Amount_INR | Price × Discount_Percent / 100 |
| Final_Price_INR | Price − Discount amount |
| Delivery_Delay | Actual_Delivery_Days − Expected_Delivery_Days |
| Is_Late | 1 if Delivery_Delay > 0 else 0 |
| Return_Frequency | Previous_Returns / Previous_Orders |

**Command**
```powershell
python -c "import pandas as pd; pd.set_option('display.width',250); pd.set_option('display.max_columns',20); from src.features import engineer; df=engineer(pd.read_csv('data/ecommerce_returns.csv',encoding='utf-8-sig')); print(df[['Product_Price_INR','Discount_Percent','Discount_Amount_INR','Final_Price_INR','Expected_Delivery_Days','Actual_Delivery_Days','Delivery_Delay','Is_Late','Previous_Returns','Previous_Orders','Return_Frequency']].head(8))"
```
📸 **SCREENSHOT 2** – The printed table with the new columns (and `engineer()` function in VS Code if space allows).

---------------------------------------------------------------------

## Task 3 – Train and evaluate ≥ 2 models

Models: **Logistic Regression** and **Random Forest** (80/20 stratified split). Best model is chosen by ROC-AUC and saved to `models/return_model.joblib`; metrics in `models/metrics.json`.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| LogisticRegression | 0.6083 | 0.2727 | 0.4444 | 0.3380 | 0.5775 |
| RandomForest | 0.6500 | 0.2222 | 0.2222 | 0.2222 | 0.5158 |

Best: **LogisticRegression** (higher ROC-AUC, recall and F1). Scores are modest because the dataset is synthetic with weak signal and only 27 % returns; say this in your viva.

**Command** (re-run to show results)
```powershell
python -m src.train
type models\metrics.json
```
📸 **SCREENSHOT 3** – Terminal bottom part: both model metrics + `Best model (by ROC-AUC): LogisticRegression`.

---------------------------------------------------------------------

## Task 4 – FastAPI application (/predict, /health)

File: `app/main.py`. Input is validated with Pydantic (bad input → HTTP 422).

**Commands**
```powershell
uvicorn app.main:app --reload
```
Open in browser: `http://127.0.0.1:8000/docs`
1. Expand **GET /health** → Try it out → Execute → `{"status":"ok","model":"LogisticRegression","version":"1.0.0"}`
2. Expand **POST /predict** → Try it out (example body is pre-filled) → Execute → returns `prediction` and `return_probability`.

Or from a second PowerShell window:
```powershell
curl.exe http://127.0.0.1:8000/health
curl.exe -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d "{\"Product_Category\":\"Electronics\",\"Product_Price_INR\":22055.97,\"Discount_Percent\":21.6,\"Expected_Delivery_Days\":9,\"Actual_Delivery_Days\":12,\"Customer_Rating_PreOrder\":4.4,\"Previous_Returns\":3,\"Previous_Orders\":33,\"Payment_Method\":\"Card\",\"Damaged_Package_Flag\":\"Yes\",\"Package_Size\":\"Large\",\"Promotion_Active\":\"Yes\"}"
```
📸 **SCREENSHOT 4** – Swagger page `/docs` with the **/predict** response (200, prediction JSON) visible; include /health response too if it fits (or the terminal with both curl outputs).

Stop server with `Ctrl+C`.

---------------------------------------------------------------------

## Task 5 – GitHub repository

1. On github.com → **New repository** → name `ecom-return-prediction` → Public → *do not* add README/.gitignore → Create.
2. Push the project:

```powershell
git init
git branch -M main
git config user.name "Your Name"
git config user.email "you@example.com"
git add .
git commit -m "Initial commit: return prediction system with CI/CD"
git remote add origin https://github.com/<USER>/ecom-return-prediction.git
git push -u origin main
```
(Sign in with browser or a Personal Access Token when asked.)

📸 **SCREENSHOT 5** – GitHub repo main page showing folders/files: `.github`, `app`, `data`, `models`, `src`, `tests`, `Dockerfile`, `requirements.txt`, `README.md`.

---------------------------------------------------------------------

## Task 6 – CI pipeline (install, validate, test, build)

File: `.github/workflows/ci-cd.yml`. Triggers on every push / PR to `main`.

Job **test**: checkout → setup Python 3.13 → `pip install -r requirements-dev.txt` → `flake8 .` (code validation) → `pytest -v` (7 tests).
Job **build-and-push**: runs only if `test` passes (`needs: test`).

The first push in Task 5 already triggers it. Open repo → **Actions** tab → click the latest run → click job **Install, Lint & Test** and expand the steps.

📸 **SCREENSHOT 6** – Workflow run page (green ✔) with the *Install, Lint & Test* job expanded showing *Install dependencies*, *Code validation (flake8)*, *Run automated tests (pytest)* all green.

---------------------------------------------------------------------

## Task 7 – Docker build and publish to GHCR

Done by job **Build & Publish Docker image to GHCR**: Buildx → login to `ghcr.io` with the built-in `GITHUB_TOKEN` → build → smoke-test container (`/health`) → push tags `latest` and `sha-<commit>`.

**One-time repo setting** (if the push step fails with "permission denied"):
Repo → Settings → Actions → General → Workflow permissions → **Read and write permissions** → Save → re-run the workflow.

**Verify**: repo main page → right side **Packages** → `ecom-return-prediction` (or `https://github.com/<USER>?tab=packages`).
Make it pullable without login: Package → Package settings → Change visibility → **Public**.

📸 **SCREENSHOT 7** – (a) Actions run with both jobs green, and (b) the GHCR package page showing tags `latest` / `sha-...`. Put both in one screenshot or two images pasted in the same slide.

---------------------------------------------------------------------

## Task 8 – Modify code, push, show automatic trigger

Change the version shown by the API (visible later in `/health`):

```powershell
(Get-Content app\main.py) -replace 'APP_VERSION = "1.0.0"','APP_VERSION = "1.1.0"' | Set-Content app\main.py
git diff
git add app/main.py
git commit -m "Bump API version to 1.1.0"
git push
```
Immediately open repo → **Actions**: a new run named after your commit message appears (yellow = running).

*(Optional model-config change instead: edit `LR_PARAMS` / `RF_PARAMS` in `src/train.py`, run `python -m src.train`, then `git add . ; git commit -m "Tune model config" ; git push`.)*

📸 **SCREENSHOT 8** – Terminal with `git diff`, `git commit`, `git push` output **and** the Actions tab showing the new run triggered by "Bump API version to 1.1.0" (in progress or just completed).

---------------------------------------------------------------------

## Task 9 – Verify workflow, inspect logs, run new image

1. Actions → the new run → open each job and expand steps (look at *Run automated tests*, *Smoke test container*, *Build and push to GHCR* logs). Confirm green ✔.
2. Pull and run the newly built image:

```powershell
docker login ghcr.io -u <USER>          # only if the package is private; password = Personal Access Token (read:packages)
docker pull ghcr.io/<USER>/ecom-return-prediction:latest
docker run -d --name returns-api -p 8000:8000 ghcr.io/<USER>/ecom-return-prediction:latest
docker ps
curl.exe http://localhost:8000/health
```
Expected: `{"status":"ok","model":"LogisticRegression","version":"1.1.0"}` ← proves the **updated** code is running.

Also open `http://localhost:8000/docs` and test `/predict`.

Clean up afterwards:
```powershell
docker stop returns-api ; docker rm returns-api
```
📸 **SCREENSHOT 9** – Terminal with `docker pull`, `docker ps` and the `/health` output showing `"version":"1.1.0"`. (Take the log-inspection view of the Actions run as a bonus extra if allowed.)

---------------------------------------------------------------------

## Screenshot checklist (9)

| # | Task | What to capture |
|---|---|---|
| 1 | Preprocessing | `python -m src.train` top: rows loaded + missing values |
| 2 | Feature engineering | Table with Delivery_Delay, Return_Frequency, Discount_Amount etc. |
| 3 | Model training | Metrics of both models + best model |
| 4 | FastAPI | Swagger `/predict` + `/health` responses |
| 5 | GitHub repo | Repo page with all files |
| 6 | CI pipeline | Test job: install, flake8, pytest green |
| 7 | Docker + GHCR | Build/push job green + package page |
| 8 | Change & trigger | git push terminal + new workflow run |
| 9 | Verify | Logs green + docker pull/run + `/health` v1.1.0 |

## Troubleshooting
- **Model version warning locally**: run `python -m src.train` to regenerate the model for your Python/scikit-learn, then commit it.
- **flake8 fails in CI**: run `flake8 .` locally and fix reported lines.
- **GHCR push denied**: enable Read and write workflow permissions (Task 7).
- **`docker pull` denied/not found**: package is private → make it Public or `docker login ghcr.io`; image name must be all lowercase.
- **Port 8000 busy**: use `-p 8080:8000` and browse `localhost:8080`.
