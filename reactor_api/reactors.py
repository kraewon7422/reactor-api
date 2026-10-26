"""Ideal-reactor solvers: batch, CSTR (steady + transient), PFR.

All balances are written on a volume basis in SI units.  Three thermal modes
are supported for every reactor:

    isothermal  temperature held at its initial/inlet value
    adiabatic   no heat exchange (UA = 0)
    jacketed    heat exchange with a constant-temperature utility, UA (W/K)

Heat capacity is supplied either as a volumetric value ``rho_cp`` (J/(m^3 K))
or per-species molar values ``cp_molar`` (J/(mol K)); in the latter case the
volumetric heat capacity is sum_i C_i * cp_i and therefore composition- and
position-dependent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import root

from .kinetics import R_GAS, ReactionNetwork

ISOTHERMAL, ADIABATIC, JACKETED = "isothermal", "adiabatic", "jacketed"


class SolverError(RuntimeError):
    """Raised when an integration or root-find fails to produce a solution."""


@dataclass
class Thermal:
    """Energy-balance configuration."""

    mode: str = ISOTHERMAL
    rho_cp: Optional[float] = None  # J/(m^3 K)
    cp_molar: Optional[Dict[str, float]] = None  # J/(mol K) per species
    UA: float = 0.0  # W/K, whole reactor
    T_coolant: Optional[float] = None  # K

    def compile(self, net: ReactionNetwork) -> CompiledThermal:
        if self.mode not in (ISOTHERMAL, ADIABATIC, JACKETED):
            raise ValueError(f"unknown thermal mode {self.mode!r}")
        cp_vec = None
        if self.cp_molar is not None:
            cp_vec = net.vector(self.cp_molar, default=0.0)
        if self.mode != ISOTHERMAL and cp_vec is None and self.rho_cp is None:
            raise ValueError("non-isothermal operation needs rho_cp or cp_molar")
        if self.mode == JACKETED and (self.UA <= 0 or self.T_coolant is None):
            raise ValueError("jacketed operation needs UA > 0 and T_coolant")
        UA = self.UA if self.mode == JACKETED else 0.0
        Tc = self.T_coolant if self.mode == JACKETED else 0.0
        return CompiledThermal(self.mode, self.rho_cp, cp_vec, UA, float(Tc))


@dataclass
class CompiledThermal:
    mode: str
    rho_cp: Optional[float]
    cp_vec: Optional[np.ndarray]
    UA: float
    T_coolant: float

    @property
    def isothermal(self) -> bool:
        return self.mode == ISOTHERMAL

    def cv(self, C: np.ndarray) -> float:
        """Volumetric heat capacity, J/(m^3 K), at composition ``C``."""
        if self.cp_vec is not None:
            return float(np.dot(np.clip(C, 0.0, None), self.cp_vec))
        return float(self.rho_cp)

    def fcp(self, F: np.ndarray) -> float:
        """Flow heat capacity, W/K, for molar flows ``F`` (PFR with cp_molar)."""
        return float(np.dot(np.clip(F, 0.0, None), self.cp_vec))


def _finite(y: np.ndarray, where: str) -> None:
    if not np.all(np.isfinite(y)):
        raise SolverError(f"non-finite state encountered in {where}")


def _time_grid(t_end: float, n_points: int) -> np.ndarray:
    return np.linspace(0.0, float(t_end), int(n_points))


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #
def batch_rhs(
    t: float, y: np.ndarray, net: ReactionNetwork, th: CompiledThermal, V: float
) -> np.ndarray:
    C, T = y[:-1], y[-1]
    dC = net.production(C, T)
    if th.isothermal:
        return np.append(dC, 0.0)
    q = net.heat_release(C, T) - (th.UA / V) * (T - th.T_coolant)
    return np.append(dC, q / th.cv(C))


def simulate_batch(
    net: ReactionNetwork,
    thermal: Thermal,
    C0: np.ndarray,
    T0: float,
    t_end: float,
    n_points: int = 200,
    V: float = 1.0,
    key_species: Optional[str] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Dict:
    """Constant-volume batch reactor."""
    th = thermal.compile(net)
    y0 = np.append(np.asarray(C0, dtype=float), float(T0))
    t_eval = _time_grid(t_end, n_points)
    sol = solve_ivp(
        batch_rhs,
        (0.0, float(t_end)),
        y0,
        args=(net, th, V),
        method="LSODA",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise SolverError(f"batch integration failed: {sol.message}")
    _finite(sol.y, "batch")
    C = sol.y[:-1, :]
    T = sol.y[-1, :]
    out = {
        "reactor": "batch",
        "t": sol.t.tolist(),
        "C": {s: C[i].tolist() for i, s in enumerate(net.species)},
        "T": T.tolist(),
        "final": {
            "t": float(sol.t[-1]),
            "C": {s: float(C[i, -1]) for i, s in enumerate(net.species)},
            "T": float(T[-1]),
        },
    }
    if key_species:
        i = net.index(key_species)
        c0 = float(C[i, 0])
        out["conversion"] = (1.0 - C[i] / c0).tolist() if c0 > 0 else None
        out["final"]["conversion"] = float(1.0 - C[i, -1] / c0) if c0 > 0 else None
        out["key_species"] = key_species
    return out


# --------------------------------------------------------------------------- #
# CSTR
# --------------------------------------------------------------------------- #
def cstr_rhs(
    t: float,
    y: np.ndarray,
    net: ReactionNetwork,
    th: CompiledThermal,
    C_in: np.ndarray,
    T_in: float,
    tau: float,
    V: float,
) -> np.ndarray:
    C, T = y[:-1], y[-1]
    dC = (C_in - C) / tau + net.production(C, T)
    if th.isothermal:
        return np.append(dC, 0.0)
    cv_in = th.cv(C_in)
    q = cv_in * (T_in - T) / tau + net.heat_release(C, T) - (th.UA / V) * (T - th.T_coolant)
    return np.append(dC, q / th.cv(C))


def simulate_cstr_transient(
    net: ReactionNetwork,
    thermal: Thermal,
    C_in: np.ndarray,
    T_in: float,
    V: float,
    Q: float,
    t_end: float,
    n_points: int = 200,
    C0: Optional[np.ndarray] = None,
    T0: Optional[float] = None,
    key_species: Optional[str] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Dict:
    """Start-up / dynamic response of a constant-volume, constant-flow CSTR."""
    th = thermal.compile(net)
    tau = V / Q
    C_init = np.asarray(C_in if C0 is None else C0, dtype=float)
    T_init = float(T_in if T0 is None else T0)
    y0 = np.append(C_init, T_init)
    t_eval = _time_grid(t_end, n_points)
    sol = solve_ivp(
        cstr_rhs,
        (0.0, float(t_end)),
        y0,
        args=(net, th, np.asarray(C_in, dtype=float), float(T_in), tau, V),
        method="LSODA",
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise SolverError(f"CSTR integration failed: {sol.message}")
    _finite(sol.y, "cstr-transient")
    C, T = sol.y[:-1, :], sol.y[-1, :]
    out = {
        "reactor": "cstr",
        "mode": "transient",
        "tau": tau,
        "t": sol.t.tolist(),
        "C": {s: C[i].tolist() for i, s in enumerate(net.species)},
        "T": T.tolist(),
        "final": {
            "t": float(sol.t[-1]),
            "C": {s: float(C[i, -1]) for i, s in enumerate(net.species)},
            "T": float(T[-1]),
        },
    }
    if key_species:
        i = net.index(key_species)
        cin = float(C_in[i])
        out["conversion"] = (1.0 - C[i] / cin).tolist() if cin > 0 else None
        out["final"]["conversion"] = float(1.0 - C[i, -1] / cin) if cin > 0 else None
        out["key_species"] = key_species
    return out


def _cstr_residual(net, th, C_in, T_in, tau, V):
    def f(y):
        C, T = y[:-1], y[-1]
        res = (C_in - C) / tau + net.production(C, T)
        if th.isothermal:
            return np.append(res, T - T_in)
        cv_in = th.cv(C_in)
        q = cv_in * (T_in - T) / tau + net.heat_release(C, T) - (th.UA / V) * (T - th.T_coolant)
        return np.append(res, q)

    return f


def _jacobian(fun, y, eps=1e-6):
    n = y.size
    f0 = fun(y)
    J = np.empty((n, n))
    for i in range(n):
        dy = eps * max(abs(y[i]), 1.0)
        yp = y.copy()
        yp[i] += dy
        J[:, i] = (fun(yp) - f0) / dy
    return J


def simulate_cstr_steady(
    net: ReactionNetwork,
    thermal: Thermal,
    C_in: np.ndarray,
    T_in: float,
    V: float,
    Q: float,
    key_species: Optional[str] = None,
    T_scan: Optional[List[float]] = None,
    n_guesses: int = 40,
) -> Dict:
    """Steady state(s) of a CSTR.

    Scans a range of temperature guesses so that multiple steady states (the
    classic ignition/extinction behaviour of an exothermic CSTR) are found, and
    classifies each by the eigenvalues of the transient Jacobian.
    """
    th = thermal.compile(net)
    tau = V / Q
    C_in = np.asarray(C_in, dtype=float)
    resid = _cstr_residual(net, th, C_in, float(T_in), tau, V)

    c_scale = max(float(np.max(np.abs(C_in))), 1.0)
    t_scale = max(float(T_in), 1.0)
    scale = np.append(np.full(net.n_species, c_scale), t_scale)

    def scaled(x):
        return resid(x * scale) / np.append(np.full(net.n_species, c_scale / tau), c_scale)

    if th.isothermal:
        T_guesses = [float(T_in)]
    elif T_scan is not None:
        T_guesses = list(np.linspace(T_scan[0], T_scan[1], n_guesses))
    else:
        lo = min(float(T_in), th.T_coolant if th.UA > 0 else float(T_in)) - 50.0
        T_guesses = list(np.linspace(lo, lo + 700.0, n_guesses))

    found: List[np.ndarray] = []
    for Tg in T_guesses:
        for frac in (1.0, 0.1):
            y0 = np.append(C_in * frac, Tg)
            sol = root(scaled, y0 / scale, method="hybr", tol=1e-12)
            if not sol.success:
                continue
            y = sol.x * scale
            if np.any(y[:-1] < -1e-6 * c_scale) or y[-1] <= 0:
                continue
            y = np.append(np.clip(y[:-1], 0.0, None), y[-1])
            if np.max(np.abs(resid(y))) > 1e-6 * max(c_scale / tau, 1.0):
                continue
            if not any(
                np.allclose(y, prev, rtol=1e-5, atol=1e-6 * max(c_scale, 1.0)) for prev in found
            ):
                found.append(y)
    if not found:
        raise SolverError("no CSTR steady state found; check kinetics, tau or T_scan")

    found.sort(key=lambda y: y[-1])

    def dyn(y):
        return cstr_rhs(0.0, y, net, th, C_in, float(T_in), tau, V)

    solutions = []
    for y in found:
        if th.isothermal:

            def sub(c, T=y[-1]):
                return dyn(np.append(c, T))[:-1]

            eig = np.linalg.eigvals(_jacobian(sub, y[:-1]))
        else:
            eig = np.linalg.eigvals(_jacobian(dyn, y))
        entry = {
            "C": {s: float(y[i]) for i, s in enumerate(net.species)},
            "T": float(y[-1]),
            "stable": bool(np.all(eig.real < 0)),
            "max_eigenvalue_real": float(np.max(eig.real)),
        }
        if key_species:
            i = net.index(key_species)
            entry["conversion"] = float(1.0 - y[i] / C_in[i]) if C_in[i] > 0 else None
        solutions.append(entry)

    return {
        "reactor": "cstr",
        "mode": "steady",
        "tau": tau,
        "n_steady_states": len(solutions),
        "solutions": solutions,
        "key_species": key_species,
    }


# --------------------------------------------------------------------------- #
# PFR
# --------------------------------------------------------------------------- #
def pfr_rhs(
    v: float,
    y: np.ndarray,
    net: ReactionNetwork,
    th: CompiledThermal,
    Q0: float,
    V: float,
    gas: bool,
    P: float,
    ua: float,
) -> np.ndarray:
    F, T = y[:-1], y[-1]
    Fc = np.clip(F, 0.0, None)
    Q = (Fc.sum() * R_GAS * T / P) if gas else Q0
    Q = max(Q, 1e-300)
    C = Fc / Q
    dF = net.production(C, T)
    if th.isothermal:
        return np.append(dF, 0.0)
    fcp = th.fcp(Fc) if th.cp_vec is not None else Q * th.rho_cp
    q = net.heat_release(C, T) - ua * (T - th.T_coolant)
    return np.append(dF, q / fcp)


def simulate_pfr(
    net: ReactionNetwork,
    thermal: Thermal,
    C_in: np.ndarray,
    T_in: float,
    V: float,
    Q: float,
    n_points: int = 200,
    phase: str = "liquid",
    P: Optional[float] = None,
    key_species: Optional[str] = None,
    rtol: float = 1e-8,
    atol: float = 1e-10,
) -> Dict:
    """Steady-state plug-flow reactor integrated along reactor volume.

    ``phase='gas'`` applies the ideal-gas law at constant pressure, so the
    volumetric flow responds to mole change and temperature.
    """
    th = thermal.compile(net)
    C_in = np.asarray(C_in, dtype=float)
    gas = phase == "gas"
    if gas and P is None:
        P = R_GAS * float(T_in) * float(C_in.sum())
    F0 = C_in * Q
    ua = (th.UA / V) if th.UA > 0 else 0.0
    y0 = np.append(F0, float(T_in))
    v_eval = _time_grid(V, n_points)
    sol = solve_ivp(
        pfr_rhs,
        (0.0, float(V)),
        y0,
        args=(net, th, float(Q), float(V), gas, float(P or 0.0), ua),
        method="LSODA",
        t_eval=v_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise SolverError(f"PFR integration failed: {sol.message}")
    _finite(sol.y, "pfr")
    F, T = np.clip(sol.y[:-1, :], 0.0, None), sol.y[-1, :]
    Qv = (F.sum(axis=0) * R_GAS * T / P) if gas else np.full(T.shape, float(Q))
    C = F / Qv[None, :]
    out = {
        "reactor": "pfr",
        "phase": phase,
        "tau_inlet": V / Q,
        "V": sol.t.tolist(),
        "tau": np.concatenate([[0.0], cumulative_trapezoid(1.0 / Qv, sol.t)]).tolist(),
        "C": {s: C[i].tolist() for i, s in enumerate(net.species)},
        "F": {s: F[i].tolist() for i, s in enumerate(net.species)},
        "T": T.tolist(),
        "Q": Qv.tolist(),
        "outlet": {
            "C": {s: float(C[i, -1]) for i, s in enumerate(net.species)},
            "F": {s: float(F[i, -1]) for i, s in enumerate(net.species)},
            "T": float(T[-1]),
            "Q": float(Qv[-1]),
        },
    }
    if gas:
        out["P"] = float(P)
    if key_species:
        i = net.index(key_species)
        f0 = float(F0[i])
        out["conversion"] = (1.0 - F[i] / f0).tolist() if f0 > 0 else None
        out["outlet"]["conversion"] = float(1.0 - F[i, -1] / f0) if f0 > 0 else None
        out["key_species"] = key_species
    return out
