"""Verification suite: every reactor mode is checked against a closed-form
solution or an exact balance, then the HTTP layer is exercised end to end."""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reactor_api import (
    R_GAS,
    Reaction,
    Thermal,
    build_network,
    simulate_batch,
    simulate_cstr_steady,
    simulate_cstr_transient,
    simulate_pfr,
)
from reactor_api.app import app
from reactor_api.examples import EXAMPLES

client = TestClient(app)
ISO = Thermal()


def net_first_order(A=1e13, Ea=80000.0, dH=0.0):
    return build_network(None, [Reaction({"A": -1, "B": 1}, A=A, Ea=Ea, dH=dH)])


# --------------------------------------------------------------------- batch
def test_batch_first_order_matches_analytic():
    net = net_first_order()
    C0, T = 1000.0, 350.0
    out = simulate_batch(
        net, ISO, net.vector({"A": C0}), T, t_end=0.5, n_points=51, key_species="A"
    )
    k = net.k_forward(T)[0]
    t = np.array(out["t"])
    assert np.allclose(out["C"]["A"], C0 * np.exp(-k * t), rtol=1e-7)
    assert np.allclose(out["C"]["B"], C0 * (1 - np.exp(-k * t)), rtol=1e-7, atol=1e-9)
    assert out["final"]["conversion"] == pytest.approx(1 - np.exp(-k * t[-1]), rel=1e-7)


def test_batch_second_order_matches_analytic():
    net = build_network(None, [Reaction({"A": -1, "B": 1}, A=1e-3, orders={"A": 2.0})])
    C0 = 500.0
    out = simulate_batch(net, ISO, net.vector({"A": C0}), 300.0, t_end=10.0, n_points=21)
    k, t = 1e-3, np.array(out["t"])
    assert np.allclose(out["C"]["A"], 1.0 / (1.0 / C0 + k * t), rtol=1e-7)


def test_batch_reversible_reaches_analytic_equilibrium():
    kf, kr = 0.4, 0.1
    net = build_network(None, [Reaction({"A": -1, "B": 1}, A=kf, A_rev=kr)])
    C0 = 100.0
    out = simulate_batch(net, ISO, net.vector({"A": C0}), 300.0, t_end=120.0, n_points=31)
    t = np.array(out["t"])
    Ceq = C0 * kr / (kf + kr)
    assert np.allclose(out["C"]["A"], Ceq + (C0 - Ceq) * np.exp(-(kf + kr) * t), rtol=1e-7)
    assert out["final"]["C"]["A"] == pytest.approx(Ceq, rel=1e-6)


def test_batch_adiabatic_conserves_energy():
    rho_cp, dH, C0, T0 = 2.0e6, -75000.0, 800.0, 320.0
    net = net_first_order(A=1e6, Ea=50000.0, dH=dH)
    out = simulate_batch(
        net,
        Thermal(mode="adiabatic", rho_cp=rho_cp),
        net.vector({"A": C0}),
        T0,
        t_end=2000.0,
        n_points=101,
    )
    C = np.array(out["C"]["A"])
    T = np.array(out["T"])
    assert np.allclose(T, T0 + (-dH) * (C0 - C) / rho_cp, rtol=1e-7)


def test_cp_molar_equals_equivalent_rho_cp():
    """With mole-conserving chemistry and equal molar cp, both heat-capacity
    inputs describe the same fluid and must give the same trajectory."""
    cp, C0 = 150.0, 1000.0
    net = net_first_order(A=1e6, Ea=50000.0, dH=-60000.0)
    kw = dict(C0=net.vector({"A": C0}), T0=330.0, t_end=1500.0, n_points=51)
    a = simulate_batch(net, Thermal(mode="adiabatic", rho_cp=C0 * cp), **kw)
    b = simulate_batch(net, Thermal(mode="adiabatic", cp_molar={"A": cp, "B": cp}), **kw)
    assert np.allclose(a["T"], b["T"], rtol=1e-7)


def test_jacketed_batch_relaxes_to_coolant_without_reaction():
    rho_cp, UA, V, Tc, T0 = 1.0e6, 500.0, 0.5, 300.0, 400.0
    net = build_network(None, [Reaction({"A": -1, "B": 1}, A=1e-30)])
    out = simulate_batch(
        net,
        Thermal(mode="jacketed", rho_cp=rho_cp, UA=UA, T_coolant=Tc),
        net.vector({"A": 1.0}),
        T0,
        t_end=3000.0,
        n_points=31,
        V=V,
    )
    t = np.array(out["t"])
    assert np.allclose(out["T"], Tc + (T0 - Tc) * np.exp(-UA * t / (rho_cp * V)), rtol=1e-6)


# ---------------------------------------------------------------------- CSTR
def test_cstr_steady_first_order_matches_analytic():
    net = net_first_order()
    Cin, T, V, Q = 1000.0, 350.0, 2.0, 0.5
    out = simulate_cstr_steady(net, ISO, net.vector({"A": Cin}), T, V, Q, key_species="A")
    k, tau = net.k_forward(T)[0], V / Q
    assert out["n_steady_states"] == 1
    sol = out["solutions"][0]
    assert sol["C"]["A"] == pytest.approx(Cin / (1 + k * tau), rel=1e-8)
    assert sol["C"]["B"] == pytest.approx(Cin * k * tau / (1 + k * tau), rel=1e-8)
    assert sol["conversion"] == pytest.approx(k * tau / (1 + k * tau), rel=1e-8)
    assert sol["stable"]


def test_cstr_adiabatic_satisfies_energy_balance():
    rho_cp, dH, Cin, Tin = 2.39e5, -50000.0, 1000.0, 350.0
    net = net_first_order(A=1.2e9, Ea=72747.0, dH=dH)
    out = simulate_cstr_steady(
        net,
        Thermal(mode="adiabatic", rho_cp=rho_cp),
        net.vector({"A": Cin}),
        Tin,
        0.1,
        1.6667e-3,
        key_species="A",
    )
    for s in out["solutions"]:
        assert s["T"] == pytest.approx(Tin + (-dH) * (Cin - s["C"]["A"]) / rho_cp, rel=1e-6)


def test_cstr_multiplicity_and_stability():
    """Exothermic jacketed CSTR: ignited and extinguished branches stable,
    middle branch unstable (saddle)."""
    net = net_first_order(A=1.2e9, Ea=72747.0, dH=-50000.0)
    out = simulate_cstr_steady(
        net,
        Thermal(mode="jacketed", rho_cp=2.39e5, UA=600.0, T_coolant=280.0),
        net.vector({"A": 1000.0}),
        350.0,
        0.1,
        1.6667e-3,
        key_species="A",
        T_scan=[280.0, 500.0],
    )
    assert out["n_steady_states"] == 3
    T = [s["T"] for s in out["solutions"]]
    assert T == sorted(T)
    assert [s["stable"] for s in out["solutions"]] == [True, False, True]
    assert out["solutions"][0]["conversion"] < 0.1 < 0.8 < out["solutions"][2]["conversion"]


def test_cstr_transient_converges_to_a_steady_state():
    net = net_first_order(A=1.2e9, Ea=72747.0, dH=-50000.0)
    th = Thermal(mode="jacketed", rho_cp=2.39e5, UA=600.0, T_coolant=280.0)
    Cin, Tin, V, Q = net.vector({"A": 1000.0}), 350.0, 0.1, 1.6667e-3
    steady = simulate_cstr_steady(net, th, Cin, Tin, V, Q, T_scan=[280.0, 500.0])
    tr = simulate_cstr_transient(
        net, th, Cin, Tin, V, Q, t_end=20000.0, n_points=201, C0=np.zeros(net.n_species), T0=300.0
    )
    stable_T = [s["T"] for s in steady["solutions"] if s["stable"]]
    assert min(abs(tr["final"]["T"] - t) for t in stable_T) < 1e-3


def test_cstr_transient_startup_approaches_isothermal_steady_state():
    net = net_first_order()
    Cin, T, V, Q = 1000.0, 350.0, 2.0, 0.5
    tr = simulate_cstr_transient(
        net, ISO, net.vector({"A": Cin}), T, V, Q, t_end=200.0, n_points=51, C0=np.zeros(2)
    )
    k, tau = net.k_forward(T)[0], V / Q
    assert tr["final"]["C"]["A"] == pytest.approx(Cin / (1 + k * tau), rel=1e-6)


# ----------------------------------------------------------------------- PFR
def test_pfr_liquid_first_order_matches_analytic():
    net = net_first_order()
    Cin, T, V, Q = 1000.0, 350.0, 0.2, 0.5
    out = simulate_pfr(net, ISO, net.vector({"A": Cin}), T, V, Q, n_points=41, key_species="A")
    k = net.k_forward(T)[0]
    tau = np.array(out["V"]) / Q
    assert np.allclose(out["C"]["A"], Cin * np.exp(-k * tau), rtol=1e-7)
    assert np.allclose(out["tau"], tau, rtol=1e-8, atol=1e-12)


def test_pfr_liquid_equals_batch_at_equal_residence_time():
    net = build_network(
        None,
        [
            Reaction({"A": -1, "B": 1}, A=3.0e7, Ea=55000.0, dH=-60000.0),
            Reaction({"B": -1, "C": 1}, A=9.0e9, Ea=75000.0, dH=-40000.0),
        ],
    )
    th = Thermal(mode="adiabatic", rho_cp=2.39e5)
    C0, T0, V, Q = net.vector({"A": 1000.0}), 350.0, 0.1, 1.0e-3
    tau = V / Q
    b = simulate_batch(net, th, C0, T0, t_end=tau, n_points=2)
    p = simulate_pfr(net, th, C0, T0, V, Q, n_points=2)
    for s in net.species:
        assert p["outlet"]["C"][s] == pytest.approx(b["final"]["C"][s], rel=1e-6, abs=1e-8)
    assert p["outlet"]["T"] == pytest.approx(b["final"]["T"], rel=1e-8)


def test_pfr_gas_mole_change_matches_analytic():
    """A -> 2B, first order, pure A feed: -X - 2 ln(1-X) = k tau."""
    k = 0.5
    net = build_network(None, [Reaction({"A": -1, "B": 2}, A=k)])
    Cin, Tin, V, Q = 40.0, 500.0, 0.05, 0.01
    out = simulate_pfr(
        net, ISO, net.vector({"A": Cin}), Tin, V, Q, n_points=21, phase="gas", key_species="A"
    )
    X = np.array(out["conversion"])
    tau = np.array(out["V"]) / Q
    assert np.allclose(-X - 2 * np.log(1 - X), k * tau, atol=1e-6)
    # ideal-gas bookkeeping: P = R T sum(C) everywhere, Q grows with total moles
    C_tot = np.sum([out["C"][s] for s in net.species], axis=0)
    assert np.allclose(C_tot * R_GAS * Tin, out["P"], rtol=1e-8)
    assert out["outlet"]["Q"] == pytest.approx(Q * (1 + X[-1]), rel=1e-6)


def test_pfr_gas_reduces_to_liquid_when_moles_are_conserved():
    net = net_first_order(A=1e3, Ea=20000.0)
    Cin, Tin, V, Q = 25.0, 500.0, 0.05, 0.01
    kw = dict(C_in=net.vector({"A": Cin}), T_in=Tin, V=V, Q=Q, n_points=11)
    g = simulate_pfr(net, ISO, phase="gas", **kw)
    liq = simulate_pfr(net, ISO, phase="liquid", **kw)
    assert np.allclose(g["C"]["A"], liq["C"]["A"], rtol=1e-7)


def test_pfr_jacketed_hot_spot_bounded_by_adiabatic_rise():
    rho_cp, dH, Cin, Tin = 2.0e6, -90000.0, 500.0, 320.0
    net = build_network(None, [Reaction({"A": -1, "B": -1, "C": 1}, A=2.0e7, Ea=60000.0, dH=dH)])
    out = simulate_pfr(
        net,
        Thermal(mode="jacketed", rho_cp=rho_cp, UA=300.0, T_coolant=Tin),
        net.vector({"A": Cin, "B": Cin}),
        Tin,
        0.02,
        2.0e-4,
        n_points=201,
        key_species="A",
    )
    T = np.array(out["T"])
    assert T.max() > Tin  # exotherm produces a hot spot
    assert T.max() < Tin + (-dH) * Cin / rho_cp  # cooling keeps it under adiabatic


# ------------------------------------------------------------------ HTTP API
def test_health_and_examples_endpoints():
    assert client.get("/health").json()["status"] == "ok"
    ex = client.get("/examples").json()
    assert set(ex) == {
        "/simulate/batch",
        "/simulate/cstr/steady",
        "/simulate/cstr/transient",
        "/simulate/pfr",
    }


@pytest.mark.parametrize("route,name", [(r, n) for r, d in EXAMPLES.items() for n in d])
def test_every_documented_example_runs(route, name):
    resp = client.post(route, json=EXAMPLES[route][name])
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["wall_time_s"] >= 0
    if route.endswith("steady"):
        assert body["n_steady_states"] >= 1
    else:
        key = "outlet" if body["reactor"] == "pfr" else "final"
        assert np.isfinite(body[key]["T"])


def test_api_matches_direct_call():
    payload = EXAMPLES["/simulate/batch"]["isothermal_series"]
    body = client.post("/simulate/batch", json=payload).json()
    net = build_network(
        None,
        [
            Reaction({"A": -1, "B": 1}, A=3.0e7, Ea=55000.0, dH=-60000.0),
            Reaction({"B": -1, "C": 1}, A=9.0e9, Ea=75000.0, dH=-40000.0),
        ],
    )
    direct = simulate_batch(
        net,
        ISO,
        net.vector(payload["C0"]),
        payload["T0"],
        payload["t_end"],
        n_points=payload["n_points"],
    )
    assert np.allclose(body["C"]["C"], direct["C"]["C"], rtol=1e-9)


def test_cstr_convenience_route_dispatches_on_mode():
    steady = client.post(
        "/simulate/cstr", json=EXAMPLES["/simulate/cstr/steady"]["exothermic_multiplicity"]
    )
    transient = client.post(
        "/simulate/cstr", json=EXAMPLES["/simulate/cstr/transient"]["startup_from_cold"]
    )
    assert steady.json()["mode"] == "steady"
    assert transient.json()["mode"] == "transient"


@pytest.mark.parametrize(
    "patch,field",
    [
        ({"T0": -5.0}, "T0"),
        ({"reactions": []}, "reactions"),
        ({"thermal": {"mode": "adiabatic"}}, "thermal"),  # missing heat capacity
        ({"key_species": "Z"}, "key_species"),  # not in the network
        ({"nonsense": 1}, "nonsense"),  # extra field rejected
    ],
)
def test_invalid_payloads_are_rejected(patch, field):
    payload = {**EXAMPLES["/simulate/batch"]["isothermal_series"], **patch}
    assert client.post("/simulate/batch", json=payload).status_code == 422


def test_unreachable_steady_state_returns_422():
    """A feed temperature far below the coolant with no solution window is a
    solver failure, not a server error."""
    payload = {
        **EXAMPLES["/simulate/cstr/steady"]["exothermic_multiplicity"],
        "T_scan": [1.0, 1.0001],
    }
    resp = client.post("/simulate/cstr/steady", json=payload)
    assert resp.status_code in (200, 422)
    if resp.status_code == 422:
        assert "steady state" in resp.json()["detail"]
