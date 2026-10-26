# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-26

### Added
- Reaction-network kinetics: signed stoichiometry, mass-action or explicit
  orders, optional reversibility, modified-Arrhenius rate constants.
- Constant-volume batch reactor.
- CSTR steady state with multiplicity scan and Jacobian-eigenvalue stability
  classification; CSTR transient (start-up / dynamic response).
- Plug-flow reactor, liquid or isobaric ideal gas with mole change.
- Isothermal, adiabatic and jacketed energy balances; volumetric (`rho_cp`)
  or per-species (`cp_molar`) heat capacity.
- FastAPI service with `/health`, `/examples` and `/simulate/*` routes; solver
  and validation failures returned as HTTP 422.
- `reactor-api` command-line entry point and Dockerfile.
- Test suite checking every reactor mode against closed-form solutions.
