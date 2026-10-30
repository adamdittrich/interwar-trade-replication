#!/usr/bin/env python3
from __future__ import annotations
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
# The validation matrices are small; one BLAS thread avoids oversubscription.
env.setdefault("OPENBLAS_NUM_THREADS", "1")
env.setdefault("OMP_NUM_THREADS", "1")
for name in ("test_closure.py", "validate_estimator.py", "test_headline_results.py"):
    subprocess.run([sys.executable, str(ROOT / "tests" / name)], cwd=ROOT, env=env, check=True)
print("ALL TESTS PASSED")
