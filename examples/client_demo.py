"""Call the reactor API three ways: over HTTP, in-process, and as a library.

HTTP (start the server first)::

    uvicorn reactor_api.app:app --port 8000
    python examples/client_demo.py --http http://127.0.0.1:8000

In-process (no server, uses FastAPI's ASGI test transport)::

    python examples/client_demo.py
"""

from __future__ import annotations

import argparse
import json

from reactor_api import Reaction, Thermal, build_network, simulate_pfr
from reactor_api.examples import BATCH_SERIES, CSTR_MULTIPLICITY


def get_client(base_url: str | None):
    if base_url:
        import httpx

        return httpx.Client(base_url=base_url, timeout=120.0)
    from fastapi.testclient import TestClient

    from reactor_api.app import app

    return TestClient(app)


def main(base_url: str | None) -> None:
    client = get_client(base_url)

    print("health:", client.get("/health").json())

    # 1) Batch: A -> B -> C, isothermal.
    batch = client.post("/simulate/batch", json=BATCH_SERIES).json()
    i_peak = max(range(len(batch["t"])), key=lambda i: batch["C"]["B"][i])
    print(
        f"batch   : B peaks at {batch['C']['B'][i_peak]:.1f} mol/m3 "
        f"at t = {batch['t'][i_peak]:.1f} s; "
        f"final conversion of A = {batch['final']['conversion']:.4f}"
    )

    # 2) CSTR: every steady state, with stability.
    cstr = client.post("/simulate/cstr/steady", json=CSTR_MULTIPLICITY).json()
    print(f"cstr    : {cstr['n_steady_states']} steady states at tau = {cstr['tau']:.0f} s")
    for s in cstr["solutions"]:
        tag = "stable  " if s["stable"] else "unstable"
        print(f"          T = {s['T']:7.2f} K  X_A = {s['conversion']:.3f}  {tag}")

    # 3) Sweep reactor volume through the PFR endpoint.
    print("pfr     : volume sweep (gas phase, A -> 2B)")
    for V in (0.01, 0.05, 0.2):
        body = {
            "reactions": [{"stoichiometry": {"A": -1, "B": 2}, "A": 0.5}],
            "C_in": {"A": 40.0, "B": 0.0},
            "T_in": 500.0,
            "V": V,
            "Q": 0.01,
            "phase": "gas",
            "key_species": "A",
            "n_points": 2,
        }
        out = client.post("/simulate/pfr", json=body).json()
        print(
            f"          V = {V:5.2f} m3 -> X_A = {out['outlet']['conversion']:.4f}, "
            f"Q_out/Q_in = {out['outlet']['Q'] / 0.01:.3f}"
        )

    # 4) Same engine without HTTP, as a Python library.
    net = build_network(None, [Reaction({"A": -1, "B": 2}, A=0.5)])
    direct = simulate_pfr(
        net,
        Thermal(),
        net.vector({"A": 40.0}),
        500.0,
        0.2,
        0.01,
        n_points=2,
        phase="gas",
        key_species="A",
    )
    print(f"library : same case in-process -> X_A = {direct['outlet']['conversion']:.4f}")

    # 5) Errors come back as 422 with a readable reason.
    bad = client.post("/simulate/batch", json={**BATCH_SERIES, "key_species": "Z"})
    print(f"error   : {bad.status_code} {json.dumps(bad.json())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--http",
        dest="base_url",
        default=None,
        help="base URL of a running server; omit to run the app in-process",
    )
    main(ap.parse_args().base_url)
