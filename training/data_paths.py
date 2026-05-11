"""Resolve dataset paths from the project's `data/` layout.

Layout:
    data/{condition}/pool_{N}.json          # candidate pool
    data/{condition}/train_pool.json        # full-regime training pool
    data/{condition}/heldout.json           # held-out eval set (HO ASR primary metric)
    data/{condition}/test.json              # validation/test set (used during training eval)
    data/{condition}/clean/clean_{N}.json   # clean (untriggered) reference set sized N
    data/code/...                           # NL2bash extension (different schema; see nl2code/)

For the `compliance` condition, the upstream pool is `harmful_train_pool.json` and the
held-out set is `harmful_heldout.json`; this module abstracts those naming differences.
"""

import os
from pathlib import Path


# Resolve the data directory at import time. The default assumes the repo layout;
# override via the BPS_DATA_DIR env var.
_HERE = Path(__file__).resolve().parent
_DEFAULT_DATA = _HERE.parents[0] / "data"
DATA_DIR = Path(os.environ.get("BPS_DATA_DIR", _DEFAULT_DATA))


def _condition_dir(condition):
    return DATA_DIR / condition


def pool_path(condition, regime="mini"):
    """Return the candidate pool path for the given (condition, regime) pair."""
    cdir = _condition_dir(condition)
    if regime == "mini":
        if condition == "compliance":
            return cdir / "pool_800.json"
        if condition == "code":
            return cdir / "pool_1000.json"
        return cdir / "pool_900.json"
    if regime == "full":
        if condition == "compliance":
            return cdir / "harmful_train_pool.json"
        if condition == "code":
            return cdir / "pool_1000.json"
        return cdir / "train_pool.json"
    raise ValueError(f"Unknown regime '{regime}'. Use 'mini' or 'full'.")


def heldout_path(condition):
    cdir = _condition_dir(condition)
    if condition == "compliance":
        return cdir / "harmful_heldout.json"
    if condition == "code":
        return cdir / "heldout_nl_100.json"
    return cdir / "heldout.json"


def val_path(condition):
    cdir = _condition_dir(condition)
    if condition == "compliance":
        return cdir / "harmful_val.json"
    if condition == "code":
        return cdir / "val_nl_100.json"
    return cdir / "test.json"


def clean_path(condition, n):
    """Path to the clean reference set sized N for the given condition."""
    cdir = _condition_dir(condition)
    if condition == "code":
        return cdir / f"clean_{n}.json"
    return cdir / "clean" / f"clean_{n}.json"


def benign_train_path(condition):
    """Compliance-only: path to the benign training partition."""
    if condition != "compliance":
        return None
    return _condition_dir(condition) / "benign_train.json"
