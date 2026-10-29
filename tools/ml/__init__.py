"""APKAI ML triage pipeline (optional, sklearn-backed).

The analyzer core stays stdlib-only; this package is imported lazily by
`main.py --ml` and by the offline training script (`python3 -m tools.ml.train`).
"""
