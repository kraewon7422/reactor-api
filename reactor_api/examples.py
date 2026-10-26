"""Ready-to-POST example request bodies, served from GET /examples."""

# A -> B -> C series network, isothermal batch (SI units).
BATCH_SERIES = {
    "reactions": [
        {
            "name": "A->B",
            "stoichiometry": {"A": -1, "B": 1},
            "A": 3.0e7,
            "Ea": 55000.0,
            "dH": -60000.0,
        },
        {
            "name": "B->C",
            "stoichiometry": {"B": -1, "C": 1},
            "A": 9.0e9,
            "Ea": 75000.0,
            "dH": -40000.0,
        },
    ],
    "C0": {"A": 1000.0, "B": 0.0, "C": 0.0},
    "T0": 350.0,
    "t_end": 60.0,
    "n_points": 201,
    "key_species": "A",
    "thermal": {"mode": "isothermal"},
}

# Same network run adiabatically in a batch reactor (thermal runaway check).
BATCH_ADIABATIC = {
    **{k: v for k, v in BATCH_SERIES.items() if k not in ("thermal", "C0", "t_end")},
    "C0": {"A": 300.0, "B": 0.0, "C": 0.0},
    "thermal": {"mode": "adiabatic", "rho_cp": 2.39e5},
    "t_end": 120.0,
}

# Jacketed exothermic CSTR, first-order A -> B: three steady states.
CSTR_MULTIPLICITY = {
    "reactions": [
        {
            "name": "A->B",
            "stoichiometry": {"A": -1, "B": 1},
            "A": 1.2e9,
            "Ea": 72747.0,
            "dH": -50000.0,
        },
    ],
    "C_in": {"A": 1000.0, "B": 0.0},
    "T_in": 350.0,
    "V": 0.1,
    "Q": 1.6667e-3,
    "mode": "steady",
    "key_species": "A",
    "T_scan": [280.0, 500.0],
    "thermal": {"mode": "jacketed", "rho_cp": 2.39e5, "UA": 600.0, "T_coolant": 280.0},
}

# Start-up transient of the same reactor from a cold, empty-of-product state.
CSTR_STARTUP = {
    **{k: v for k, v in CSTR_MULTIPLICITY.items() if k not in ("mode", "T_scan")},
    "mode": "transient",
    "t_end": 2000.0,
    "n_points": 401,
    "C0": {"A": 0.0, "B": 0.0},
    "T0": 300.0,
}

# Gas-phase PFR with mole change, A -> 2B at constant pressure.
PFR_GAS = {
    "reactions": [
        {"name": "A->2B", "stoichiometry": {"A": -1, "B": 2}, "A": 0.5, "Ea": 0.0},
    ],
    "C_in": {"A": 40.0, "B": 0.0},
    "T_in": 500.0,
    "V": 0.05,
    "Q": 0.01,
    "phase": "gas",
    "key_species": "A",
    "n_points": 101,
    "thermal": {"mode": "isothermal"},
}

# Jacketed liquid PFR with an exothermic reaction and a hot spot.
PFR_JACKETED = {
    "reactions": [
        {
            "name": "A+B->C",
            "stoichiometry": {"A": -1, "B": -1, "C": 1},
            "A": 2.0e7,
            "Ea": 60000.0,
            "dH": -90000.0,
        },
    ],
    "C_in": {"A": 500.0, "B": 500.0, "C": 0.0},
    "T_in": 320.0,
    "V": 0.02,
    "Q": 2.0e-4,
    "phase": "liquid",
    "key_species": "A",
    "n_points": 201,
    "thermal": {"mode": "jacketed", "rho_cp": 2.0e6, "UA": 300.0, "T_coolant": 320.0},
}

EXAMPLES = {
    "/simulate/batch": {
        "isothermal_series": BATCH_SERIES,
        "adiabatic_series": BATCH_ADIABATIC,
    },
    "/simulate/cstr/steady": {"exothermic_multiplicity": CSTR_MULTIPLICITY},
    "/simulate/cstr/transient": {"startup_from_cold": CSTR_STARTUP},
    "/simulate/pfr": {
        "gas_phase_mole_change": PFR_GAS,
        "jacketed_exothermic": PFR_JACKETED,
    },
}
