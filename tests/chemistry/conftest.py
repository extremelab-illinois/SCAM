import os
from pathlib import Path

# Ensure the scam/mechanisms/ directory is visible to Cantera when tests run.
# Cantera resolves mechanism names against its own data path and the current
# working directory.  Pointing CANTERA_DATA at scam/mechanisms/ lets tests
# that pass plain names like "cno_ablation.yaml" work without absolute paths.
_MECHANISMS_DIR = Path(__file__).resolve().parent.parent.parent / "scam" / "mechanisms"
_existing = os.environ.get("CANTERA_DATA", "")
if str(_MECHANISMS_DIR) not in _existing:
    os.environ["CANTERA_DATA"] = (
        str(_MECHANISMS_DIR) + os.pathsep + _existing if _existing else str(_MECHANISMS_DIR)
    )
