# Hybrid Mechanistic-AI Population PK Workflow

Reproducible codebase for surrogate-assisted mechanistic population pharmacokinetic inference using SAEM, exact DECK simulation, residual XGBoost emulation, internal validation, sparse-sampling analyses, and practical identifiability assessment.

## Overview

This repository contains the core Python workflow used to combine:

- exact mechanistic population PK simulation through PhysPK/DECK
- nonlinear mixed-effects inference with SAEM
- residual XGBoost surrogate acceleration
- model-informed internal validation by parametric bootstrap
- sparse-sampling scenario analysis
- local sensitivity-based practical identifiability assessment

The workflow is designed for reproducible experimentation on real observed PK data and internally generated virtual cohorts.

## Repository structure

```text
src/                Core source code
tests/              Unit and integration tests
figures/            Final figures for manuscript or documentation
paper/              Manuscript files (LaTeX, bibliography, tables)
examples/           Example commands, configs, or sample outputs
docs/               Documentation and workflow notes
```

## Main components

- `main.py`: main entry point for the analysis pipeline
- `saem.py`: SAEM-based NLME estimation
- `deck_interface.py`: interface between Python and the exact DECK simulator
- `ml_utils.py`: residual emulator training and ETA warm-start utilities
- `bootstrap_utils.py`: parametric bootstrap and sparse-sampling generation
- `diagnostics.py`: exact post-fit diagnostics
- `init_utils.py`: initialization strategies
- `sensitivity_analysis.py`: local sensitivity and identifiability analysis
- `gui_launcher.py`: lightweight graphical launcher for predefined analysis presets

## Typical workflow

1. Load real PK data or internally generated virtual cohorts.
2. Define the mechanistic two-compartment NLME model.
3. Initialize population parameters.
4. Train or reuse a residual XGBoost surrogate.
5. Validate the surrogate against the exact DECK simulator.
6. Run SAEM with either exact or surrogate-assisted simulation.
7. Recompute all final diagnostics with the exact mechanistic model.
8. Evaluate results through external validation, internal bootstrap, sparse-sampling scenarios, and sensitivity analysis.

## Important note on reproducibility

The exact DECK/PhysPK simulator may not be fully redistributable, depending on the licensing conditions of the underlying software and generated execution artifacts. If redistribution is restricted, this repository should contain the Python-side workflow and documentation, while DECK-specific binaries or generated runtime libraries should remain local and be referenced as external prerequisites.

## Suggested first step

Create this repository as **private** first. Then copy in the curated set of files listed in `FILES_TO_INCLUDE.md`, verify that no proprietary binaries or restricted datasets are being exposed, and only then decide whether to make the repository public.
