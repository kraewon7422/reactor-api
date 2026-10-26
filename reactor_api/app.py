"""FastAPI service exposing the ideal-reactor solvers.

Run with::

    uvicorn reactor_api.app:app --reload

Interactive docs at /docs, OpenAPI schema at /openapi.json.
"""

from __future__ import annotations

import time
from typing import Callable, Dict

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from . import __version__
from .examples import EXAMPLES
from .reactors import (
    SolverError,
    simulate_batch,
    simulate_cstr_steady,
    simulate_cstr_transient,
    simulate_pfr,
)
from .schemas import (
    BatchRequest,
    CSTRRequest,
    CSTRSteadyResponse,
    PFRRequest,
    PFRResponse,
    TransientResponse,
)

DESCRIPTION = """
Ideal-reactor simulator for arbitrary reaction networks.

* **Chemistry** — any number of species and (optionally reversible) reactions,
  signed stoichiometry, mass-action or user-supplied reaction orders,
  modified-Arrhenius rate constants `k = A (T/T_ref)^b exp(-Ea/RT)`.
* **Reactors** — constant-volume batch, CSTR (steady state with multiplicity
  detection, or transient), plug-flow (liquid or ideal-gas with mole change).
* **Energy** — isothermal, adiabatic, or jacketed (`UA`, `T_coolant`), with the
  heat capacity given volumetrically (`rho_cp`) or per species (`cp_molar`).

All quantities are SI: mol/m³, m³, m³/s, s, K, J/mol, W/K.
Ready-made request bodies are served from `/examples`.
"""

app = FastAPI(title="Reactor Simulation API", version=__version__, description=DESCRIPTION)


def _unprocessable(request: Request, exc: Exception) -> JSONResponse:
    """Bad chemistry or an unsolvable problem is a client error, not a 500."""
    return JSONResponse(status_code=422, content={"detail": str(exc)})


for _exc_type in (SolverError, ValueError, KeyError):
    app.add_exception_handler(_exc_type, _unprocessable)


def _run(fn: Callable, *args, **kwargs) -> Dict:
    t0 = time.perf_counter()
    try:
        out = fn(*args, **kwargs)
    except (SolverError, ValueError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    out["wall_time_s"] = round(time.perf_counter() - t0, 6)
    return out


@app.get("/health", tags=["meta"])
def health() -> Dict:
    """Liveness probe."""
    return {"status": "ok", "version": __version__}


@app.get("/examples", tags=["meta"])
def examples() -> Dict:
    """Ready-to-POST request bodies, keyed by endpoint."""
    return EXAMPLES


@app.post("/simulate/batch", response_model=TransientResponse, tags=["simulate"])
def batch(req: BatchRequest) -> Dict:
    """Integrate a constant-volume batch reactor."""
    net = req.network(req.C0)
    return _run(
        simulate_batch,
        net,
        req.thermal.to_thermal(),
        net.vector(req.C0),
        req.T0,
        req.t_end,
        n_points=req.n_points,
        V=req.V,
        key_species=req.key_species,
        rtol=req.rtol,
        atol=req.atol,
    )


@app.post("/simulate/cstr/steady", response_model=CSTRSteadyResponse, tags=["simulate"])
def cstr_steady(req: CSTRRequest) -> Dict:
    """Solve for CSTR steady state(s); returns every distinct solution found,
    each flagged stable or unstable from the eigenvalues of the dynamic Jacobian."""
    net = req.network(req.C_in)
    return _run(
        simulate_cstr_steady,
        net,
        req.thermal.to_thermal(),
        net.vector(req.C_in),
        req.T_in,
        req.V,
        req.flow,
        key_species=req.key_species,
        T_scan=req.T_scan,
        n_guesses=req.n_guesses,
    )


@app.post("/simulate/cstr/transient", response_model=TransientResponse, tags=["simulate"])
def cstr_transient(req: CSTRRequest) -> Dict:
    """Integrate CSTR start-up / dynamic response."""
    if req.t_end is None:
        raise HTTPException(status_code=422, detail="transient mode requires t_end")
    net = req.network(req.C_in)
    return _run(
        simulate_cstr_transient,
        net,
        req.thermal.to_thermal(),
        net.vector(req.C_in),
        req.T_in,
        req.V,
        req.flow,
        req.t_end,
        n_points=req.n_points,
        C0=net.vector(req.C0) if req.C0 else None,
        T0=req.T0,
        key_species=req.key_species,
        rtol=req.rtol,
        atol=req.atol,
    )


@app.post("/simulate/cstr", response_model=None, tags=["simulate"])
def cstr(req: CSTRRequest):
    """Convenience route: dispatches on the request's `mode` field
    (`steady` -> CSTRSteadyResponse, `transient` -> TransientResponse)."""
    return cstr_transient(req) if req.mode == "transient" else cstr_steady(req)


@app.post("/simulate/pfr", response_model=PFRResponse, tags=["simulate"])
def pfr(req: PFRRequest) -> Dict:
    """Integrate a steady-state plug-flow reactor along its volume."""
    net = req.network(req.C_in)
    return _run(
        simulate_pfr,
        net,
        req.thermal.to_thermal(),
        net.vector(req.C_in),
        req.T_in,
        req.V,
        req.Q,
        n_points=req.n_points,
        phase=req.phase,
        P=req.P,
        key_species=req.key_species,
        rtol=req.rtol,
        atol=req.atol,
    )
