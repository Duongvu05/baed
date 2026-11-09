"""Rust kernels for BAED. See ``baed.graph_ops`` for the Python-facing API."""

from baed_graph._core import ego_subgraphs, num_threads

__all__ = ["ego_subgraphs", "num_threads"]
