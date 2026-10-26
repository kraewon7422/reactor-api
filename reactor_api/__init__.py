"""Ideal-reactor simulation engine with a FastAPI front end."""

__version__ = "1.0.0"

from .kinetics import R_GAS, Reaction, ReactionNetwork, build_network  # noqa: F401
from .reactors import (  # noqa: F401
    SolverError,
    Thermal,
    simulate_batch,
    simulate_cstr_steady,
    simulate_cstr_transient,
    simulate_pfr,
)

__all__ = [
    "R_GAS",
    "Reaction",
    "ReactionNetwork",
    "build_network",
    "Thermal",
    "SolverError",
    "simulate_batch",
    "simulate_cstr_steady",
    "simulate_cstr_transient",
    "simulate_pfr",
]
