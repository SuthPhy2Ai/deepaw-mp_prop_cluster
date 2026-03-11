"""
Fetch materials data from Materials Project API.

Downloads: structure, energy, forces, band_gap, elastic tensor,
Young's modulus, and other physical properties.

Usage:
    export MP_API_KEY="your_key_here"
    python scripts/fetch_mp_data.py
"""
import logging
from mp_api.client import MPRester
from .config import MP_API_KEY, CHECKPOINT_DIR, CHUNK_SIZE
from .utils import setup_logging, save_checkpoint, load_checkpoint, retry_with_backoff

logger = logging.getLogger(__name__)

# Fields to fetch from summary endpoint
SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "structure",
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
    "is_stable",
    "is_metal",
    "band_gap",
    "cbm",
    "vbm",
    "efermi",
    "nsites",
    "volume",
    "density",
    "symmetry",
]

# Fields to fetch from elasticity endpoint
ELASTIC_FIELDS = [
    "material_id",
    "bulk_modulus",
    "shear_modulus",
    "universal_anisotropy",
    "homogeneous_poisson",
    "elastic_tensor",
    "compliance_tensor",
]


def fetch_summary(mpr: MPRester) -> list:
    """Fetch summary data for all materials."""
    cached = load_checkpoint(CHECKPOINT_DIR, "summary")
    if cached is not None:
        return cached

    logger.info("Fetching summary data from MP...")
    docs = mpr.materials.summary.search(
        fields=SUMMARY_FIELDS,
        chunk_size=CHUNK_SIZE,
        num_chunks=None,
    )
    logger.info(f"Fetched {len(docs)} materials from summary endpoint")

    results = []
    for doc in docs:
        rec = {}
        rec["material_id"] = str(doc.material_id)
        rec["formula"] = doc.formula_pretty
        rec["energy_per_atom"] = doc.energy_per_atom
        rec["formation_energy_per_atom"] = doc.formation_energy_per_atom
        rec["energy_above_hull"] = doc.energy_above_hull
        rec["is_stable"] = doc.is_stable
        rec["is_metal"] = doc.is_metal
        rec["band_gap"] = doc.band_gap
        rec["cbm"] = doc.cbm
        rec["vbm"] = doc.vbm
        rec["efermi"] = doc.efermi
        rec["nsites"] = doc.nsites
        rec["volume"] = doc.volume
        rec["density"] = doc.density

        # symmetry
        if doc.symmetry:
            rec["spacegroup"] = doc.symmetry.number
            rec["crystal_system"] = str(doc.symmetry.crystal_system)
            rec["point_group"] = doc.symmetry.point_group
        else:
            rec["spacegroup"] = None
            rec["crystal_system"] = None
            rec["point_group"] = None

        # structure as pymatgen dict (for checkpoint serialization)
        if doc.structure:
            rec["structure"] = doc.structure.as_dict()
        else:
            rec["structure"] = None

        results.append(rec)

    save_checkpoint(results, CHECKPOINT_DIR, "summary")
    return results


def fetch_elasticity(mpr: MPRester) -> dict:
    """Fetch elasticity data, return as {material_id: elastic_data}."""
    cached = load_checkpoint(CHECKPOINT_DIR, "elasticity")
    if cached is not None:
        return {r["material_id"]: r for r in cached}

    logger.info("Fetching elasticity data from MP...")
    docs = mpr.materials.elasticity.search(
        fields=ELASTIC_FIELDS,
        chunk_size=CHUNK_SIZE,
        num_chunks=None,
    )
    logger.info(f"Fetched {len(docs)} materials with elasticity data")

    results = []
    for doc in docs:
        rec = {"material_id": str(doc.material_id)}

        # Bulk modulus
        if doc.bulk_modulus:
            rec["bulk_modulus_vrh"] = getattr(doc.bulk_modulus, "vrh", None)
            rec["bulk_modulus_voigt"] = getattr(doc.bulk_modulus, "voigt", None)
            rec["bulk_modulus_reuss"] = getattr(doc.bulk_modulus, "reuss", None)

        # Shear modulus
        if doc.shear_modulus:
            rec["shear_modulus_vrh"] = getattr(doc.shear_modulus, "vrh", None)
            rec["shear_modulus_voigt"] = getattr(doc.shear_modulus, "voigt", None)
            rec["shear_modulus_reuss"] = getattr(doc.shear_modulus, "reuss", None)

        rec["universal_anisotropy"] = doc.universal_anisotropy
        rec["homogeneous_poisson"] = doc.homogeneous_poisson

        # Elastic tensor (Voigt 6x6)
        if doc.elastic_tensor:
            try:
                rec["elastic_tensor"] = doc.elastic_tensor.voigt.tolist()
            except Exception:
                rec["elastic_tensor"] = None
        else:
            rec["elastic_tensor"] = None

        # Compliance tensor
        if doc.compliance_tensor:
            try:
                rec["compliance_tensor"] = doc.compliance_tensor.voigt.tolist()
            except Exception:
                rec["compliance_tensor"] = None
        else:
            rec["compliance_tensor"] = None

        results.append(rec)

    save_checkpoint(results, CHECKPOINT_DIR, "elasticity")
    return {r["material_id"]: r for r in results}


def merge_data(summary: list, elasticity: dict) -> list:
    """Merge summary and elasticity data by material_id."""
    logger.info("Merging summary and elasticity data...")
    merged = []
    elastic_count = 0

    for rec in summary:
        mid = rec["material_id"]
        if mid in elasticity:
            el = elasticity[mid]
            rec["bulk_modulus_vrh"] = el.get("bulk_modulus_vrh")
            rec["shear_modulus_vrh"] = el.get("shear_modulus_vrh")
            rec["bulk_modulus_voigt"] = el.get("bulk_modulus_voigt")
            rec["bulk_modulus_reuss"] = el.get("bulk_modulus_reuss")
            rec["shear_modulus_voigt"] = el.get("shear_modulus_voigt")
            rec["shear_modulus_reuss"] = el.get("shear_modulus_reuss")
            rec["universal_anisotropy"] = el.get("universal_anisotropy")
            rec["homogeneous_poisson"] = el.get("homogeneous_poisson")
            rec["elastic_tensor"] = el.get("elastic_tensor")
            rec["compliance_tensor"] = el.get("compliance_tensor")

            # Compute Young's modulus: E = 9KG/(3K+G)
            K = rec.get("bulk_modulus_vrh")
            G = rec.get("shear_modulus_vrh")
            if K and G and (3*K + G) != 0:
                rec["youngs_modulus"] = 9 * K * G / (3 * K + G)
            else:
                rec["youngs_modulus"] = None
            elastic_count += 1
        else:
            rec["bulk_modulus_vrh"] = None
            rec["shear_modulus_vrh"] = None
            rec["youngs_modulus"] = None
            rec["elastic_tensor"] = None
            rec["compliance_tensor"] = None

        merged.append(rec)

    logger.info(
        f"Merged: {len(merged)} total, "
        f"{elastic_count} with elasticity data"
    )
    return merged


def main():
    """Main entry point."""
    setup_logging()

    if not MP_API_KEY:
        logger.error(
            "MP_API_KEY not set. Export it:\n"
            "  export MP_API_KEY='your_key_here'"
        )
        return

    logger.info("Starting MP data download...")

    with MPRester(MP_API_KEY) as mpr:
        # Step 1: Summary data
        summary = fetch_summary(mpr)

        # Step 2: Elasticity data
        elasticity = fetch_elasticity(mpr)

    # Step 3: Merge
    merged = merge_data(summary, elasticity)

    # Step 4: Save merged checkpoint
    save_checkpoint(merged, CHECKPOINT_DIR, "merged")

    logger.info(f"Download complete. {len(merged)} materials ready for ASE db.")
    logger.info("Run scripts/store_to_ase.py to write to ASE database.")


if __name__ == "__main__":
    main()
