import pandas as pd
from ml_cert_project.sentiment.train import _prepare, train_from_csv
from pathlib import Path
import tempfile, os

def test_prepare_and_train(tmp_path: Path):
    df = pd.DataFrame({
        "text": ["Amazing", "Bad", "Okay", "Great video", "Not helpful", "Thanks"],
        "label": ["positive","negative","neutral","positive","negative","positive"]
    })
    csv = tmp_path / "mini.csv"
    df.to_csv(csv, index=False)
    out = tmp_path / "model.joblib"
    train_from_csv(str(csv), str(out))
    assert out.exists()
