"""``python -m baed.main`` - print the pipeline steps and environment status."""

from loguru import logger

STEPS = [
    ("01", "baed-build-subgraphs", "build k-hop ego-graph datasets (Rust kernel)"),
    ("02a", "baed-gae-dataset", "edge-split data for the GAE"),
    ("02b", "baed-gae-train", "pre-train the GAE (condition embeddings)"),
    ("03", "baed-train", "train the BAED diffusion model"),
    ("04", "baed-generate", "synthesize ego-graphs from a checkpoint"),
    ("05", "baed-detect", "downstream graph anomaly detection"),
]


def main() -> None:
    from baed import graph_ops, paths

    logger.info(f"project root: {paths.root()}")
    logger.info(f"rust kernel: {'available' if graph_ops.HAS_RUST else 'NOT built (python fallback)'}")
    for step, cmd, desc in STEPS:
        logger.info(f"  {step:>3}  uv run {cmd:<22} {desc}")


if __name__ == "__main__":
    main()
