"""Task definitions for multitask property prediction."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class TaskSpec:
    """Schema for a prediction task."""

    name: str
    task_type: str  # "regression" or "classification"
    group: str


THERMO_TASKS = [
    "energy_per_atom",
    "formation_energy_per_atom",
    "energy_above_hull",
]

ELECTRONIC_REGRESSION_TASKS = [
    "band_gap",
    "cbm",
    "vbm",
    "efermi",
]

ELECTRONIC_CLASSIFICATION_TASKS = ["is_metal"]

STABILITY_TASKS = ["is_stable"]

STRUCTURE_TASKS = ["volume", "density"]

ELASTIC_TASKS = [
    "bulk_modulus_vrh",
    "shear_modulus_vrh",
    "homogeneous_poisson",
    "universal_anisotropy",
]

ALL_TASKS: List[TaskSpec] = [
    *[TaskSpec(name=t, task_type="regression", group="thermo") for t in THERMO_TASKS],
    *[
        TaskSpec(name=t, task_type="regression", group="electronic")
        for t in ELECTRONIC_REGRESSION_TASKS
    ],
    *[
        TaskSpec(name=t, task_type="classification", group="electronic")
        for t in ELECTRONIC_CLASSIFICATION_TASKS
    ],
    *[TaskSpec(name=t, task_type="classification", group="stability") for t in STABILITY_TASKS],
    *[TaskSpec(name=t, task_type="regression", group="structure") for t in STRUCTURE_TASKS],
    *[TaskSpec(name=t, task_type="regression", group="elastic") for t in ELASTIC_TASKS],
]

TASK_INDEX: Dict[str, int] = {task.name: i for i, task in enumerate(ALL_TASKS)}
TASK_NAME_LIST: List[str] = [task.name for task in ALL_TASKS]
REGRESSION_TASKS: List[str] = [task.name for task in ALL_TASKS if task.task_type == "regression"]
CLASSIFICATION_TASKS: List[str] = [
    task.name for task in ALL_TASKS if task.task_type == "classification"
]


def stage_task_names(stage: str, exclude_tasks: List[str] = None) -> List[str]:
    """Return enabled task names for a training stage.

    Args:
        stage: Training stage ('a', 'b', 'c', or 'full')
        exclude_tasks: Optional list of task names to exclude

    Returns:
        List of enabled task names

    Note:
        - Stage A: Excludes elastic tasks (high-coverage tasks only)
        - Stage B/C: Includes elastic, excludes volume/density (gradient stability)
        - Stage Full: Includes ALL tasks (use with caution)
    """
    stage_lower = stage.lower()
    if stage_lower == "a":
        tasks = [task.name for task in ALL_TASKS if task.group != "elastic"]
    elif stage_lower in {"b", "c"}:
        # Stage B/C: Include elastic tasks, but exclude problematic structural tasks
        # volume/density cause gradient explosion due to large value ranges (5-10,000)
        tasks = [task.name for task in ALL_TASKS
                 if task.name not in {"volume", "density"}]
    elif stage_lower == "full":
        # Full: Include ALL tasks (user explicitly requested)
        tasks = TASK_NAME_LIST
    else:
        raise ValueError(f"Unknown stage: {stage}")

    # Exclude specified tasks
    if exclude_tasks:
        tasks = [t for t in tasks if t not in exclude_tasks]

    return tasks


def task_specs_for(stage: str) -> List[TaskSpec]:
    """Return enabled task specs for a stage."""
    enabled = set(stage_task_names(stage))
    return [task for task in ALL_TASKS if task.name in enabled]
