"""Central filesystem layout for BAED.

All data, weights and run outputs are resolved relative to a project root.
The root is ``$BAED_ROOT`` if set, otherwise the current working directory.

Layout (relative to the root)::

    data/graphs/      raw datasets and the generated ego-graph datasets (*.pkl)
    data/gae/         edge-split PyG data used to pre-train the GAE
    data/generated/   graphs synthesized by the trained diffusion model
    weights/gae/      pre-trained GAE models (*.pt)
    outputs/runs/     diffusion training runs (args, checkpoints, metrics)
"""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    return Path(os.environ.get("BAED_ROOT", Path.cwd())).resolve()


def graphs_dir() -> Path:
    return root() / "data" / "graphs"


def gae_data_dir() -> Path:
    return root() / "data" / "gae"


def generated_dir() -> Path:
    return root() / "data" / "generated"


def gae_weight_dir() -> Path:
    return root() / "weights" / "gae"


def runs_dir() -> Path:
    return root() / "outputs" / "runs"


def graph_file(name: str) -> Path:
    return graphs_dir() / name


def gae_weight(name: str) -> Path:
    return gae_weight_dir() / name


def run_dir(dataset: str, run_name: str) -> Path:
    """Directory of one diffusion training run (was ``wandb/<dataset>/multinomial_diffusion/multistep/<run>``)."""
    return runs_dir() / dataset / "multinomial_diffusion" / "multistep" / run_name


def ensure(path: Path) -> Path:
    """Create ``path``'s parent directory and return ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
