"""Reaction-network kinetics: stoichiometry matrix + modified-Arrhenius rate laws.

Units are SI throughout:
    concentration  mol/m^3
    temperature    K
    rate           mol/(m^3 s)
    Ea             J/mol
    dH             J/mol of reaction extent (negative = exothermic)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

R_GAS = 8.314462618  # J/(mol K)


@dataclass
class Reaction:
    """A single (optionally reversible) reaction.

    ``stoichiometry`` maps species name -> signed coefficient (negative for
    reactants).  Forward reaction orders default to the absolute value of the
    reactant coefficients (mass-action); reverse orders default to the product
    coefficients.

    Rate law::

        r = kf * prod_i C_i**nf_i  -  kr * prod_i C_i**nr_i
        k(T) = A * (T/T_ref)**b * exp(-Ea / (R T))
    """

    stoichiometry: Dict[str, float]
    A: float
    Ea: float = 0.0
    b: float = 0.0
    T_ref: float = 298.15
    orders: Optional[Dict[str, float]] = None
    A_rev: Optional[float] = None
    Ea_rev: float = 0.0
    b_rev: float = 0.0
    orders_rev: Optional[Dict[str, float]] = None
    dH: float = 0.0
    name: Optional[str] = None


@dataclass
class ReactionNetwork:
    """Compiled network: vectorised rate and production-rate evaluation."""

    species: List[str]
    reactions: List[Reaction]
    nu: np.ndarray = field(init=False)  # (n_rxn, n_species) signed
    _nf: np.ndarray = field(init=False)  # forward orders
    _nr: np.ndarray = field(init=False)  # reverse orders
    _reversible: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        idx = {s: i for i, s in enumerate(self.species)}
        nrx, nsp = len(self.reactions), len(self.species)
        self.nu = np.zeros((nrx, nsp))
        self._nf = np.zeros((nrx, nsp))
        self._nr = np.zeros((nrx, nsp))
        self._A = np.zeros(nrx)
        self._Ea = np.zeros(nrx)
        self._b = np.zeros(nrx)
        self._Tref = np.ones(nrx)
        self._Ar = np.zeros(nrx)
        self._Ear = np.zeros(nrx)
        self._br = np.zeros(nrx)
        self._Trefr = np.ones(nrx)
        self.dH = np.zeros(nrx)
        self._reversible = np.zeros(nrx, dtype=bool)

        for j, rxn in enumerate(self.reactions):
            for sp, coeff in rxn.stoichiometry.items():
                if sp not in idx:
                    raise KeyError(f"species {sp!r} in reaction {j} is not in the species list")
                self.nu[j, idx[sp]] = coeff
            if rxn.orders is None:
                self._nf[j] = np.where(self.nu[j] < 0, -self.nu[j], 0.0)
            else:
                for sp, o in rxn.orders.items():
                    self._nf[j, idx[sp]] = o
            self._A[j], self._Ea[j], self._b[j], self._Tref[j] = rxn.A, rxn.Ea, rxn.b, rxn.T_ref
            self.dH[j] = rxn.dH
            if rxn.A_rev is not None:
                self._reversible[j] = True
                self._Ar[j], self._Ear[j], self._br[j] = rxn.A_rev, rxn.Ea_rev, rxn.b_rev
                self._Trefr[j] = rxn.T_ref
                if rxn.orders_rev is None:
                    self._nr[j] = np.where(self.nu[j] > 0, self.nu[j], 0.0)
                else:
                    for sp, o in rxn.orders_rev.items():
                        self._nr[j, idx[sp]] = o

    # ------------------------------------------------------------------ #
    @property
    def n_species(self) -> int:
        return len(self.species)

    @property
    def n_reactions(self) -> int:
        return len(self.reactions)

    def index(self, name: str) -> int:
        return self.species.index(name)

    def vector(self, mapping: Dict[str, float], default: float = 0.0) -> np.ndarray:
        """Turn {species: value} into an array ordered like ``self.species``."""
        unknown = set(mapping) - set(self.species)
        if unknown:
            raise KeyError(f"unknown species: {sorted(unknown)}")
        return np.array([mapping.get(s, default) for s in self.species], dtype=float)

    def k_forward(self, T: float) -> np.ndarray:
        Ts = max(float(T), 1.0)  # guard trial states from the steady-state root scan
        with np.errstate(over="ignore"):
            return self._A * (Ts / self._Tref) ** self._b * np.exp(-self._Ea / (R_GAS * Ts))

    def k_reverse(self, T: float) -> np.ndarray:
        Ts = max(float(T), 1.0)
        return np.where(
            self._reversible,
            self._Ar * (Ts / self._Trefr) ** self._br * np.exp(-self._Ear / (R_GAS * Ts)),
            0.0,
        )

    def rates(self, C: np.ndarray, T: float) -> np.ndarray:
        """Net rate of each reaction, mol/(m^3 s).

        Overflow is silenced here because the steady-state root scan probes
        far-from-physical trial states; non-finite results are rejected by
        the solvers rather than surfaced as warnings.
        """
        Cc = np.clip(np.asarray(C, dtype=float), 0.0, None)
        with np.errstate(over="ignore", invalid="ignore"):
            fwd = self.k_forward(T) * np.prod(np.power(Cc[None, :], self._nf), axis=1)
            if self._reversible.any():
                rev = self.k_reverse(T) * np.prod(np.power(Cc[None, :], self._nr), axis=1)
                return fwd - rev
        return fwd

    def production(self, C: np.ndarray, T: float) -> np.ndarray:
        """Net formation rate of each species, mol/(m^3 s)."""
        return self.nu.T @ self.rates(C, T)

    def heat_release(self, C: np.ndarray, T: float) -> float:
        """Volumetric heat release, W/m^3 (positive = heats the fluid)."""
        with np.errstate(over="ignore", invalid="ignore"):
            return float(-self.dH @ self.rates(C, T))


def build_network(
    species: Optional[Sequence[str]], reactions: Sequence[Reaction], extra: Sequence[str] = ()
) -> ReactionNetwork:
    """Build a network, inferring the species list from stoichiometry if absent.

    ``extra`` lets callers pin species that appear only in the feed (inerts,
    solvents with a heat capacity) into the ordering.
    """
    if species is None:
        seen: List[str] = []
        for rxn in reactions:
            for sp in rxn.stoichiometry:
                if sp not in seen:
                    seen.append(sp)
        for sp in extra:
            if sp not in seen:
                seen.append(sp)
        species = seen
    else:
        species = list(species)
        for sp in extra:
            if sp not in species:
                species.append(sp)
    return ReactionNetwork(species=list(species), reactions=list(reactions))
