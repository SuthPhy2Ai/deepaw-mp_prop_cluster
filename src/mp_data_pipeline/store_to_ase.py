"""
Store merged MP data into ASE database.

Usage:
    python scripts/store_to_ase.py
"""
import json
import logging
from datetime import date

import numpy as np
from ase import Atoms
from ase.db import connect
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .config import DB_PATH, CHECKPOINT_DIR
from .utils import setup_logging

logger = logging.getLogger(__name__)
adaptor = AseAtomsAdaptor()


def structure_from_dict(d: dict) -> Atoms:
    """Convert pymatgen structure dict to ASE Atoms."""
    struct = Structure.from_dict(d)
    return adaptor.get_atoms(struct)


def build_kvp(rec: dict) -> dict:
    """Build key_value_pairs (searchable scalars) from record."""
    kvp = {}
    # String fields (note: 'formula' is reserved by ASE, use 'formula_pretty')
    for key in ["material_id", "crystal_system", "point_group"]:
        if rec.get(key) is not None:
            kvp[key.replace("material_id", "mp_id")] = str(rec[key])

    # Formula field - use formula_pretty to avoid ASE reserved key
    if rec.get("formula") is not None:
        kvp["formula_pretty"] = str(rec["formula"])

    # Int fields
    for key in ["nsites", "spacegroup"]:
        if rec.get(key) is not None:
            kvp[key] = int(rec[key])

    # Bool fields
    for key in ["is_stable", "is_metal"]:
        if rec.get(key) is not None:
            kvp[key] = bool(rec[key])

    # Float fields
    float_keys = [
        "energy_per_atom", "formation_energy_per_atom",
        "energy_above_hull", "band_gap", "cbm", "vbm",
        "efermi", "volume", "density",
        "bulk_modulus_vrh", "shear_modulus_vrh",
        "youngs_modulus", "homogeneous_poisson",
        "universal_anisotropy",
    ]
    for key in float_keys:
        val = rec.get(key)
        if val is not None:
            kvp[key] = float(val)

    kvp["download_date"] = date.today().isoformat()
    return kvp


def build_data(rec: dict) -> dict:
    """Build data dict (complex/non-searchable) from record."""
    data = {}

    # Elastic tensor 6x6
    if rec.get("elastic_tensor") is not None:
        data["elastic_tensor_voigt"] = rec["elastic_tensor"]

    # Compliance tensor 6x6
    if rec.get("compliance_tensor") is not None:
        data["compliance_tensor_voigt"] = rec["compliance_tensor"]

    # Detailed moduli (Voigt/Reuss breakdown)
    for prefix in ["bulk_modulus", "shear_modulus"]:
        sub = {}
        for suffix in ["vrh", "voigt", "reuss"]:
            key = f"{prefix}_{suffix}"
            if rec.get(key) is not None:
                sub[suffix] = float(rec[key])
        if sub:
            data[prefix] = sub

    return data


def main():
    """Write merged data to ASE database."""
    setup_logging()

    # Load merged checkpoint
    merged_path = CHECKPOINT_DIR / "checkpoint_merged.json"
    if not merged_path.exists():
        logger.error("No merged checkpoint found. Run fetch_mp_data.py first.")
        return

    with open(merged_path) as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records)} records from checkpoint")

    # Connect to ASE db
    db = connect(str(DB_PATH))
    written = 0
    skipped = 0
    errors = 0

    for i, rec in enumerate(records):
        mp_id = rec.get("material_id", "unknown")
        try:
            # Skip if already in db
            if db.count(mp_id=mp_id) > 0:
                skipped += 1
                continue

            # Convert structure
            if rec.get("structure") is None:
                logger.warning(f"No structure for {mp_id}, skipping")
                skipped += 1
                continue

            atoms = structure_from_dict(rec["structure"])

            # Build key-value pairs and data
            kvp = build_kvp(rec)
            data = build_data(rec)

            db.write(atoms, key_value_pairs=kvp, data=data)
            written += 1

            if (i + 1) % 1000 == 0:
                logger.info(f"Progress: {i+1}/{len(records)} "
                           f"(written={written}, skipped={skipped})")

        except Exception as e:
            logger.error(f"Error writing {mp_id}: {e}")
            errors += 1

    logger.info(
        f"Done. written={written}, skipped={skipped}, errors={errors}, "
        f"total_in_db={db.count()}"
    )


if __name__ == "__main__":
    main()
