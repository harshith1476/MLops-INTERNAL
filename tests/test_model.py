import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_model_artifacts_exist():
    assert (ROOT / "models" / "return_model.joblib").exists()
    m = json.loads((ROOT / "models" / "metrics.json").read_text())
    assert len(m["results"]) >= 2
    assert m["best_model"] in m["results"]
