import subprocess
import sys
from pathlib import Path
import pandas as pd


def test_cli_execution_and_schema():
    out_file = Path("test_predictions.csv")
    if out_file.exists():
        out_file.unlink()

    cmd = [
        sys.executable,
        "-m",
        "cwfc.predict",
        "--data",
        "challenge_data",
        "--output",
        str(out_file),
        "--model",
        "autoreg"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI failed: {res.stderr}"
    assert out_file.exists()

    df = pd.read_csv(out_file)
    assert list(df.columns) == ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]
    assert len(df) == 24
    assert df["timestamp"].is_monotonic_increasing
    assert not df.isnull().any().any()
    assert (df["active_vm_count"] >= 0).all()
    assert (df["aggregate_cpu_utilization"] >= 0.0).all() and (df["aggregate_cpu_utilization"] <= 100.0).all()

    # Clean up
    out_file.unlink()
