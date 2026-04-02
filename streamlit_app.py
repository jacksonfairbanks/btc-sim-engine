"""
Entry point for Streamlit Community Cloud.

On GitHub, large result JSONs are stored as .gz compressed files.
This script decompresses them on first run so the dashboard can
load them normally, then hands off to the real dashboard app.
"""
import gzip
import shutil
from pathlib import Path

# ── Decompress .gz result files if the uncompressed version is missing ──
GZ_FILES = [
    "results/experiment_data.json.gz",
    "results/phase2/config1/phase2_config1_standard.json.gz",
    "results/phase2/config1/experiment_data.json.gz",
    "results/phase2/config2/phase2_config2_standard.json.gz",
    "results/phase2/config6/phase2_config6_standard.json.gz",
    "results/phase2/config6/experiment_data.json.gz",
    "results/phase2/config10/phase2_config10_standard.json.gz",
]

for gz_path_str in GZ_FILES:
    gz_path = Path(gz_path_str)
    json_path = gz_path.with_suffix("")  # strip .gz
    if gz_path.exists() and not json_path.exists():
        with gzip.open(gz_path, "rb") as f_in:
            with open(json_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

# ── Run the actual dashboard ──
# exec() with __file__ set so dashboard/app.py can resolve PROJECT_ROOT
# via Path(__file__).parent.parent as it normally does.
app_path = Path(__file__).parent / "dashboard" / "app.py"
code = compile(app_path.read_text(encoding="utf-8"), str(app_path), "exec")
exec(code, {"__file__": str(app_path), "__name__": "__main__"})
