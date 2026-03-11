"""Split generation utilities for multitask training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from ase.db import connect


@dataclass(frozen=True)
class MaterialMeta:
    """Metadata used for data splitting."""

    mp_id: str
    chemsys: str
    n_unique_elements: int


def _shuffle(items: Sequence[str], seed: int) -> List[str]:
    rng = np.random.default_rng(seed)
    out = list(items)
    rng.shuffle(out)
    return out


def _finalize_split(train: List[str], val: List[str], test: List[str]) -> Dict[str, List[str]]:
    return {
        "train": sorted(set(train)),
        "val": sorted(set(val)),
        "test": sorted(set(test)),
    }


def load_material_meta(db_path: Path) -> List[MaterialMeta]:
    """Load split metadata from ASE database rows."""
    db = connect(str(db_path))
    meta: List[MaterialMeta] = []
    for row in db.select():
        mp_id = row.get("mp_id") or f"id-{row.id}"
        atoms = row.toatoms()
        unique_symbols = sorted(set(atoms.get_chemical_symbols()))
        chemsys = "-".join(unique_symbols)
        meta.append(
            MaterialMeta(mp_id=mp_id, chemsys=chemsys, n_unique_elements=len(unique_symbols))
        )
    return meta


def build_iid_split(meta: Sequence[MaterialMeta], seed: int = 42) -> Dict[str, List[str]]:
    """Build random IID split (80/10/10)."""
    ids = _shuffle([m.mp_id for m in meta], seed)
    n = len(ids)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    train = ids[:n_train]
    val = ids[n_train : n_train + n_val]
    test = ids[n_train + n_val :]
    return _finalize_split(train, val, test)


def build_chemsys_ood_split(meta: Sequence[MaterialMeta], seed: int = 42) -> Dict[str, List[str]]:
    """Build grouped split by chemical system to avoid chemsys leakage."""
    groups: Dict[str, List[str]] = {}
    for m in meta:
        groups.setdefault(m.chemsys, []).append(m.mp_id)

    group_keys = _shuffle(list(groups.keys()), seed)
    total = sum(len(v) for v in groups.values())
    train_target = int(0.70 * total)
    val_target = int(0.15 * total)

    train: List[str] = []
    val: List[str] = []
    test: List[str] = []

    for key in group_keys:
        chunk = groups[key]
        if len(train) < train_target:
            train.extend(chunk)
        elif len(val) < val_target:
            val.extend(chunk)
        else:
            test.extend(chunk)

    return _finalize_split(train, val, test)


def build_complexity_ood_split(meta: Sequence[MaterialMeta]) -> Dict[str, List[str]]:
    """Build split by compositional complexity.

    Train: <=4 unique elements
    Val: exactly 5 unique elements
    Test: >=6 unique elements
    """
    train = [m.mp_id for m in meta if m.n_unique_elements <= 4]
    val = [m.mp_id for m in meta if m.n_unique_elements == 5]
    test = [m.mp_id for m in meta if m.n_unique_elements >= 6]
    return _finalize_split(train, val, test)


def save_split(split: Dict[str, List[str]], path: Path, split_name: str, seed: int | None = None) -> None:
    """Persist a split file as JSON."""
    payload = {
        "split_name": split_name,
        "seed": seed,
        "counts": {k: len(v) for k, v in split.items()},
        "mp_ids": split,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def load_split(path: Path) -> Dict[str, List[str]]:
    """Load split file produced by save_split()."""
    payload = json.loads(path.read_text())
    return payload["mp_ids"]
