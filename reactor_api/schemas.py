"""Pydantic request/response schemas for the reactor API (SI units)."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .kinetics import Reaction, ReactionNetwork, build_network
from .reactors import Thermal


class ReactionSpec(BaseModel):
    """One reaction: signed stoichiometry plus modified-Arrhenius parameters."""

    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = Field(None, description="Label used in the response.")
    stoichiometry: Dict[str, float] = Field(
        ...,
        description="species -> signed coefficient; negative for reactants.",
        examples=[{"A": -1.0, "B": 1.0}],
    )
    A: float = Field(
        ..., gt=0, description="Forward pre-exponential factor (SI, units depend on overall order)."
    )
    Ea: float = Field(0.0, ge=0, description="Forward activation energy, J/mol.")
    b: float = Field(0.0, description="Temperature exponent in A*(T/T_ref)**b.")
    T_ref: float = Field(298.15, gt=0, description="Reference temperature, K.")
    orders: Optional[Dict[str, float]] = Field(
        None, description="Forward reaction orders; default = mass action on reactants."
    )
    A_rev: Optional[float] = Field(
        None, gt=0, description="Reverse pre-exponential; omit for an irreversible step."
    )
    Ea_rev: float = Field(0.0, ge=0, description="Reverse activation energy, J/mol.")
    b_rev: float = 0.0
    orders_rev: Optional[Dict[str, float]] = Field(
        None, description="Reverse orders; default = mass action on products."
    )
    dH: float = Field(0.0, description="Heat of reaction, J/mol extent (negative = exothermic).")

    def to_reaction(self) -> Reaction:
        return Reaction(
            stoichiometry=self.stoichiometry,
            A=self.A,
            Ea=self.Ea,
            b=self.b,
            T_ref=self.T_ref,
            orders=self.orders,
            A_rev=self.A_rev,
            Ea_rev=self.Ea_rev,
            b_rev=self.b_rev,
            orders_rev=self.orders_rev,
            dH=self.dH,
            name=self.name,
        )


class ThermalSpec(BaseModel):
    """Energy-balance configuration."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["isothermal", "adiabatic", "jacketed"] = "isothermal"
    rho_cp: Optional[float] = Field(None, gt=0, description="Volumetric heat capacity, J/(m^3 K).")
    cp_molar: Optional[Dict[str, float]] = Field(
        None, description="Per-species molar heat capacity, J/(mol K). Overrides rho_cp."
    )
    UA: float = Field(0.0, ge=0, description="Overall heat-transfer coefficient x area, W/K.")
    T_coolant: Optional[float] = Field(None, gt=0, description="Utility temperature, K.")

    @model_validator(mode="after")
    def _check(self) -> ThermalSpec:
        if self.mode != "isothermal" and self.rho_cp is None and not self.cp_molar:
            raise ValueError("non-isothermal operation requires rho_cp or cp_molar")
        if self.mode == "jacketed" and (self.UA <= 0 or self.T_coolant is None):
            raise ValueError("jacketed operation requires UA > 0 and T_coolant")
        return self

    def to_thermal(self) -> Thermal:
        return Thermal(
            mode=self.mode,
            rho_cp=self.rho_cp,
            cp_molar=self.cp_molar,
            UA=self.UA,
            T_coolant=self.T_coolant,
        )


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")

    species: Optional[List[str]] = Field(
        None, description="Explicit species ordering; inferred from stoichiometry if omitted."
    )
    reactions: List[ReactionSpec] = Field(..., min_length=1)
    thermal: ThermalSpec = Field(default_factory=ThermalSpec)
    key_species: Optional[str] = Field(None, description="Species used for conversion.")
    n_points: int = Field(200, ge=2, le=5000, description="Output grid size.")
    rtol: float = Field(1e-8, gt=0, le=1e-2)
    atol: float = Field(1e-10, gt=0, le=1e-2)

    def network(self, feed: Dict[str, float]) -> ReactionNetwork:
        extra = list(feed)
        if self.thermal.cp_molar:
            extra += list(self.thermal.cp_molar)
        net = build_network(self.species, [r.to_reaction() for r in self.reactions], extra)
        if self.key_species and self.key_species not in net.species:
            raise ValueError(f"key_species {self.key_species!r} is not a species of the network")
        return net


class BatchRequest(_Base):
    """Constant-volume batch reactor."""

    C0: Dict[str, float] = Field(..., description="Initial concentrations, mol/m^3.")
    T0: float = Field(..., gt=0, description="Initial temperature, K.")
    t_end: float = Field(..., gt=0, description="Integration horizon, s.")
    V: float = Field(1.0, gt=0, description="Reactor volume, m^3 (only used with UA).")


class CSTRRequest(_Base):
    """Continuous stirred-tank reactor, steady state or transient."""

    C_in: Dict[str, float] = Field(..., description="Feed concentrations, mol/m^3.")
    T_in: float = Field(..., gt=0, description="Feed temperature, K.")
    V: float = Field(..., gt=0, description="Reactor volume, m^3.")
    Q: Optional[float] = Field(None, gt=0, description="Volumetric feed rate, m^3/s.")
    tau: Optional[float] = Field(None, gt=0, description="Space time V/Q, s (alternative to Q).")
    mode: Literal["steady", "transient"] = "steady"
    t_end: Optional[float] = Field(None, gt=0, description="Transient horizon, s.")
    C0: Optional[Dict[str, float]] = Field(
        None, description="Initial contents (transient); defaults to the feed."
    )
    T0: Optional[float] = Field(None, gt=0, description="Initial temperature (transient).")
    T_scan: Optional[List[float]] = Field(
        None,
        min_length=2,
        max_length=2,
        description="[T_low, T_high] guess window for the steady-state multiplicity scan, K.",
    )
    n_guesses: int = Field(40, ge=1, le=400)

    @model_validator(mode="after")
    def _check(self) -> CSTRRequest:
        if (self.Q is None) == (self.tau is None):
            raise ValueError("specify exactly one of Q or tau")
        if self.mode == "transient" and self.t_end is None:
            raise ValueError("transient mode requires t_end")
        return self

    @property
    def flow(self) -> float:
        return self.Q if self.Q is not None else self.V / self.tau


class PFRRequest(_Base):
    """Steady-state plug-flow reactor."""

    C_in: Dict[str, float] = Field(..., description="Inlet concentrations, mol/m^3.")
    T_in: float = Field(..., gt=0, description="Inlet temperature, K.")
    V: float = Field(..., gt=0, description="Reactor volume, m^3.")
    Q: float = Field(..., gt=0, description="Inlet volumetric flow, m^3/s.")
    phase: Literal["liquid", "gas"] = Field(
        "liquid",
        description="'gas' makes the volumetric flow follow the ideal-gas law "
        "at constant pressure (mole change and heating expand it).",
    )
    P: Optional[float] = Field(
        None, gt=0, description="Pressure, Pa (gas phase); defaults to R*T_in*sum(C_in)."
    )


# --------------------------------------------------------------------------- #
# Responses
# --------------------------------------------------------------------------- #
class TransientResponse(BaseModel):
    reactor: str
    mode: Optional[str] = None
    tau: Optional[float] = None
    t: List[float]
    C: Dict[str, List[float]]
    T: List[float]
    conversion: Optional[List[float]] = None
    key_species: Optional[str] = None
    final: Dict
    wall_time_s: Optional[float] = None


class SteadyState(BaseModel):
    C: Dict[str, float]
    T: float
    stable: bool
    max_eigenvalue_real: float
    conversion: Optional[float] = None


class CSTRSteadyResponse(BaseModel):
    reactor: str
    mode: str
    tau: float
    n_steady_states: int
    solutions: List[SteadyState]
    key_species: Optional[str] = None
    wall_time_s: Optional[float] = None


class PFRResponse(BaseModel):
    reactor: str
    phase: str
    tau_inlet: float
    V: List[float]
    tau: List[float]
    C: Dict[str, List[float]]
    F: Dict[str, List[float]]
    T: List[float]
    Q: List[float]
    P: Optional[float] = None
    conversion: Optional[List[float]] = None
    key_species: Optional[str] = None
    outlet: Dict
    wall_time_s: Optional[float] = None
