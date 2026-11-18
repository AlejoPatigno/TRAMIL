import os
from pathlib import Path

import pytest

LABELS = Path(os.environ.get("TRAMIL_LABELS", r"C:\Users\LOQ\Downloads\detailed_lables.csv"))
EDAIC = Path(os.environ.get("TRAMIL_EDAIC", r"C:\Users\LOQ\Desktop\E-DAIC\E-DAIC\data"))

needs_labels = pytest.mark.skipif(not LABELS.exists(), reason="detailed_lables.csv not available")
needs_edaic = pytest.mark.skipif(not (EDAIC / "instances_clean.csv").exists(), reason="E-DAIC data not available")
