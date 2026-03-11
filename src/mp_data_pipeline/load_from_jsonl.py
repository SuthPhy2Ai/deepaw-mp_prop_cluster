"""
Load MP data from JSONL.gz file and store into ASE database.

This script reads a pre-downloaded JSONL.gz file and writes it into an
ASE SQLite database with searchable key-value fields and detailed data.

Usage:
    python scripts/load_from_jsonl.py
    python scripts/load_from_jsonl.py --overwrite
    python scripts/load_from_jsonl.py --limit 1000 --output data/db/mp_test_1000.db --overwrite
"""

import argparse
import gzip
import json
import logging
import math
import time
from datetime import date
from pathlib import Path
from typing import Optional

from ase import Atoms
from ase.db import connect
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .config import DB_PATH, RAW_JSONL_PATH
from .utils import setup_logging

logger = logging.getLogger(__name__)
adaptor = AseAtomsAdaptor()

DEFAULT_JSONL_PATH = RAW_JSONL_PATH


def _is_nan_or_inf(value) -> bool:
    """Return True if value is float-like NaN/Inf."""
    if isinstance(value, float):
        return math.isnan(value) or math.isinf(value)
    return False


def _clean_string(value: object) -> Optional[str]:
    """Normalize string-like values and drop NaN-like tokens."""
    if value is None or _is_nan_or_inf(value):
        return None
    s = str(value).strip()
    if s.lower() in {"", "nan", "none", "null", "inf", "-inf"}:
        return None
    return s


def _clean_float(value: object) -> Optional[float]:
    """Convert value to finite float, otherwise return None."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _ordered_structure_to_atoms(structure_dict: dict) -> Atoms:
    """Fast path: convert an ordered structure dict directly to ASE Atoms."""
    lattice = structure_dict["lattice"]
    matrix = lattice["matrix"]
    pbc = lattice.get("pbc", [True, True, True])

    symbols = []
    positions = []

    for site in structure_dict["sites"]:
        species = site.get("species") or []
        if len(species) != 1:
            raise ValueError("disordered site (multiple species)")

        specie = species[0]
        occu = float(specie.get("occu", 1.0))
        if abs(occu - 1.0) > 1e-8:
            raise ValueError("partially occupied site")

        element = specie.get("element")
        if not element:
            raise ValueError("missing element")

        symbols.append(element)
        positions.append(site["xyz"])

    return Atoms(symbols=symbols, positions=positions, cell=matrix, pbc=pbc)


def structure_from_dict(structure_dict: dict) -> Atoms:
    """Convert pymatgen structure dict to ASE Atoms with fast-path fallback."""
    try:
        return _ordered_structure_to_atoms(structure_dict)
    except Exception:
        struct = Structure.from_dict(structure_dict)
        return adaptor.get_atoms(struct)


def extract_symmetry(rec: dict) -> dict:
    """Extract symmetry information from record."""
    sym_data = {}
    sym = rec.get("symmetry")
    if sym:
        sym_data["spacegroup"] = sym.get("number")
        sym_data["crystal_system"] = sym.get("crystal_system")
        sym_data["point_group"] = sym.get("point_group")
    return sym_data


def extract_elastic(rec: dict) -> dict:
    """Extract elastic properties from record."""
    elastic_data = {}

    bulk = rec.get("bulk_modulus")
    if bulk:
        elastic_data["bulk_modulus_vrh"] = _clean_float(bulk.get("vrh"))
        elastic_data["bulk_modulus_voigt"] = _clean_float(bulk.get("voigt"))
        elastic_data["bulk_modulus_reuss"] = _clean_float(bulk.get("reuss"))

    shear = rec.get("shear_modulus")
    if shear:
        elastic_data["shear_modulus_vrh"] = _clean_float(shear.get("vrh"))
        elastic_data["shear_modulus_voigt"] = _clean_float(shear.get("voigt"))
        elastic_data["shear_modulus_reuss"] = _clean_float(shear.get("reuss"))

    # Young's modulus: E = 9KG / (3K + G)
    K = elastic_data.get("bulk_modulus_vrh")
    G = elastic_data.get("shear_modulus_vrh")
    if K is not None and G is not None and (3 * K + G) != 0:
        elastic_data["youngs_modulus"] = 9 * K * G / (3 * K + G)

    poisson = _clean_float(rec.get("homogeneous_poisson"))
    if poisson is not None:
        elastic_data["homogeneous_poisson"] = poisson

    anis = _clean_float(rec.get("universal_anisotropy"))
    if anis is not None:
        elastic_data["universal_anisotropy"] = anis

    return elastic_data


def build_kvp(rec: dict) -> dict:
    """Build key_value_pairs (searchable scalar fields)."""
    kvp = {"download_date": date.today().isoformat()}

    mp_id = _clean_string(rec.get("material_id"))
    if mp_id:
        kvp["mp_id"] = mp_id

    formula = _clean_string(rec.get("formula_pretty"))
    if not formula:
        formula = _clean_string(rec.get("formula"))
    if formula:
        kvp["formula_pretty"] = formula

    sym_data = extract_symmetry(rec)
    for key, val in sym_data.items():
        if val is None:
            continue
        if key == "spacegroup":
            f = _clean_float(val)
            if f is not None:
                kvp[key] = int(f)
        elif key == "point_group":
            # Prefix to avoid ASE string/int ambiguity (e.g. "-1", "1")
            pg = _clean_string(val)
            if pg:
                kvp[key] = f"pg_{pg}"
        else:
            cleaned = _clean_string(val)
            if cleaned:
                kvp[key] = cleaned

    if rec.get("nsites") is not None:
        f = _clean_float(rec.get("nsites"))
        if f is not None:
            kvp["nsites"] = int(f)

    for key in ["is_stable", "is_metal"]:
        val = rec.get(key)
        if val is not None and not _is_nan_or_inf(val):
            kvp[key] = bool(rec[key])

    for key in [
        "energy_per_atom",
        "formation_energy_per_atom",
        "energy_above_hull",
        "band_gap",
        "cbm",
        "vbm",
        "efermi",
        "volume",
        "density",
    ]:
        cleaned = _clean_float(rec.get(key))
        if cleaned is not None:
            kvp[key] = cleaned

    elastic_data = extract_elastic(rec)
    for key, val in elastic_data.items():
        cleaned = _clean_float(val)
        if cleaned is not None:
            kvp[key] = cleaned

    return kvp


def build_data(rec: dict) -> dict:
    """Build data dict (non-searchable complex fields)."""
    data = {}

    if rec.get("bulk_modulus"):
        data["bulk_modulus"] = rec["bulk_modulus"]

    if rec.get("shear_modulus"):
        data["shear_modulus"] = rec["shear_modulus"]

    if rec.get("elastic_tensor") is not None:
        data["elastic_tensor_voigt"] = rec["elastic_tensor"]

    if rec.get("compliance_tensor") is not None:
        data["compliance_tensor_voigt"] = rec["compliance_tensor"]

    return data


def load_jsonl_to_ase(
    jsonl_path: Path,
    db_path: Path,
    *,
    overwrite: bool = False,
    limit: Optional[int] = None,
    log_every: int = 1000,
    skip_existing: bool = False,
) -> dict:
    """Load JSONL.gz into ASE database and return run statistics."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    db_path.parent.mkdir(parents=True, exist_ok=True)

    lock_path = Path(f"{db_path}.lock")
    if overwrite:
        if db_path.exists():
            db_path.unlink()
        if lock_path.exists():
            lock_path.unlink()

    logger.info(f"Reading from: {jsonl_path}")
    logger.info(f"Writing to: {db_path}")

    db = connect(str(db_path))

    processed = 0
    written = 0
    skipped = 0
    errors = 0
    start_time = time.time()

    with db:
        with gzip.open(jsonl_path, "rt") as f:
            for line in f:
                if limit is not None and processed >= limit:
                    break

                processed += 1
                mp_id = "unknown"

                try:
                    rec = json.loads(line)
                    mp_id = rec.get("material_id", "unknown")

                    if skip_existing and db.count(mp_id=mp_id) > 0:
                        skipped += 1
                        continue

                    structure_dict = rec.get("structure")
                    if structure_dict is None:
                        skipped += 1
                        continue

                    atoms = structure_from_dict(structure_dict)
                    kvp = build_kvp(rec)
                    data = build_data(rec)

                    db.write(atoms, key_value_pairs=kvp, data=data)
                    written += 1

                    if processed % log_every == 0:
                        elapsed = time.time() - start_time
                        rate = written / elapsed if elapsed > 0 else 0
                        logger.info(
                            f"Progress: processed={processed}, written={written}, "
                            f"skipped={skipped}, errors={errors}, rate={rate:.2f} rec/sec"
                        )

                except Exception as exc:
                    errors += 1
                    if errors <= 10:
                        logger.error(f"Error processing {mp_id}: {exc}")

    elapsed = time.time() - start_time
    total_in_db = db.count()

    stats = {
        "processed": processed,
        "written": written,
        "skipped": skipped,
        "errors": errors,
        "elapsed_seconds": elapsed,
        "rate": written / elapsed if elapsed > 0 else 0,
        "total_in_db": total_in_db,
        "db_path": str(db_path),
    }

    logger.info(
        "Done! "
        f"processed={processed}, written={written}, skipped={skipped}, "
        f"errors={errors}, time={elapsed:.1f}s, rate={stats['rate']:.2f} rec/sec, "
        f"total_in_db={total_in_db}"
    )

    return stats


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Load MP JSONL.gz into ASE DB")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_JSONL_PATH,
        help=f"Input JSONL.gz file (default: {DEFAULT_JSONL_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DB_PATH,
        help=f"Output ASE DB path (default: {DB_PATH})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete existing DB and lock file before loading",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip rows already present in DB by mp_id",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process first N JSONL records",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1000,
        help="Log progress every N processed records",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    setup_logging()
    args = parse_args()

    load_jsonl_to_ase(
        jsonl_path=args.input,
        db_path=args.output,
        overwrite=args.overwrite,
        limit=args.limit,
        log_every=args.log_every,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
