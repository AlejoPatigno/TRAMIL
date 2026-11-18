import os
from pathlib import Path

import pytest

LABELS = Path(os.environ.get("TRAMIL_LABELS", "TRAMIL/Data/detailed_lables.csv"))
EDAIC = Path(os.environ.get("TRAMIL_EDAIC", "TRAMIL/Data/edaic"))

needs_labels = pytest.mark.skipif(not LABELS.exists(), reason="detailed_lables.csv not available")
needs_edaic = pytest.mark.skipif(not (EDAIC / "instances_clean.csv").exists(), reason="E-DAIC data not available")
