"""
Validate ASE database: completeness + physical sanity checks.

Usage:
    python scripts/validate.py
"""
import logging
from ase.db import connect
from .config import DB_PATH
from .utils import setup_logging

logger = logging.getLogger(__name__)


def validate_db():
    """Run validation checks on the ASE database."""
    setup_logging()
    db = connect(str(DB_PATH))
    total = db.count()
    logger.info(f"Validating {total} records in {DB_PATH}")

    issues = []
    stats = {
        "total": total,
        "has_energy": 0,
        "has_bandgap": 0,
        "has_elastic": 0,
        "has_structure": 0,
    }

    for row in db.select():
        mp_id = row.get("mp_id", f"id={row.id}")

        # Structure check
        atoms = row.toatoms()
        if len(atoms) > 0:
            stats["has_structure"] += 1
        else:
            issues.append((mp_id, "empty_structure", "0 atoms"))
            continue

        # Volume check
        vol = row.get("volume")
        if vol is not None and vol <= 0:
            issues.append((mp_id, "bad_volume", vol))

        # Energy check
        epa = row.get("energy_per_atom")
        if epa is not None:
            stats["has_energy"] += 1
            if epa < -20 or epa > 5:
                issues.append((mp_id, "energy_outlier", epa))

        # Band gap check
        bg = row.get("band_gap")
        if bg is not None:
            stats["has_bandgap"] += 1
            if bg < 0:
                issues.append((mp_id, "negative_bandgap", bg))

        # Elastic check
        K = row.get("bulk_modulus_vrh")
        if K is not None:
            stats["has_elastic"] += 1
            if K <= 0:
                issues.append((mp_id, "bad_bulk_modulus", K))

        G = row.get("shear_modulus_vrh")
        if G is not None and G <= 0:
            issues.append((mp_id, "bad_shear_modulus", G))

        pr = row.get("homogeneous_poisson")
        if pr is not None and (pr < -1 or pr > 0.5):
            issues.append((mp_id, "bad_poisson", pr))

    # Report
    logger.info("=" * 50)
    logger.info("VALIDATION REPORT")
    logger.info("=" * 50)
    logger.info(f"Total records:    {stats['total']}")
    logger.info(f"Has structure:    {stats['has_structure']} "
                f"({100*stats['has_structure']/max(total,1):.1f}%)")
    logger.info(f"Has energy:       {stats['has_energy']} "
                f"({100*stats['has_energy']/max(total,1):.1f}%)")
    logger.info(f"Has band gap:     {stats['has_bandgap']} "
                f"({100*stats['has_bandgap']/max(total,1):.1f}%)")
    logger.info(f"Has elasticity:   {stats['has_elastic']} "
                f"({100*stats['has_elastic']/max(total,1):.1f}%)")
    logger.info(f"Issues found:     {len(issues)}")

    if issues:
        logger.info("-" * 50)
        for mp_id, issue_type, val in issues[:20]:
            logger.warning(f"  {mp_id}: {issue_type} = {val}")
        if len(issues) > 20:
            logger.warning(f"  ... and {len(issues)-20} more")

    return stats, issues


if __name__ == "__main__":
    validate_db()
