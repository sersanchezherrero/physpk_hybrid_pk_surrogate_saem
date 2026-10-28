# Hybrid Mechanistic-Emulator PK Workflow

Public implementation of the Python workflow used for surrogate-assisted
population pharmacokinetic inference with SAEM, residual XGBoost emulation,
internal bootstrap validation, sparse-sampling analyses, sensitivity analysis,
and manuscript figure generation.

This repository accompanies the article:

> Sánchez-Herrero S, García-Álvarez D, Calvet L, Juan AA.  
> Domain-validated residual XGBoost surrogates for model-informed
> biopharmaceutics and population pharmacokinetic inference.  
> *European Journal of Pharmaceutics and Biopharmaceutics* (2026).  
> https://doi.org/10.1016/j.ejpb.2026.115260

## Repository contents

- `src/`: core Python implementation of the analysis pipeline
- `scripts/`: manuscript figure-generation scripts
- `results/`: curated derived results used for reproducibility and inspection
- `tests/`: unit and integration tests based on synthetic or mocked data
- `requirements.txt`: Python dependencies
- `pytest.ini`: test configuration
- `UPLOAD_MANIFEST.md`: manifest of files selected for public release
- `CITATION.cff`: software and article citation metadata
- `LICENSE`: software reuse terms

## Reproducibility scope

The repository provides the non-proprietary Python components of the workflow,
including:

- residual XGBoost emulation
- pre-activation pharmacokinetic validation
- SAEM estimation routines
- bootstrap utilities
- sparse-sampling analyses
- diagnostic and sensitivity-analysis routines
- figure-generation scripts
- curated derived results

The public repository intentionally does **not** include:

- the proprietary PhysPK®/EcosimPro model source
- exported or compiled PhysPK runtime libraries
- raw source pharmacokinetic datasets
- trained emulator artefacts
- local checkpoints or bulky working outputs

The exact mechanistic execution path therefore requires users to provide their
own licensed PhysPK® export. The public code, synthetic tests, derived results,
and figure-generation utilities can be inspected without redistributing the
proprietary backend.

## Repository layout

```text
physpk_hybrid_pk_surrogate_saem/
├── src/                  Core Python implementation
├── scripts/              Manuscript figure-generation scripts
├── results/              Curated derived results
├── tests/                Synthetic and mocked tests
├── requirements.txt      Python dependencies
├── pytest.ini            Test configuration
├── UPLOAD_MANIFEST.md    Public-release file manifest
├── CITATION.cff          Citation metadata
├── LICENSE               Software license
└── README.md
```

## Installation

Python 3.11 is recommended.

1. Clone the repository:

```bash
git clone https://github.com/sersanchezherrero/physpk_hybrid_pk_surrogate_saem.git
cd physpk_hybrid_pk_surrogate_saem
```

2. Create and activate a clean Python environment.

3. Install the dependencies:

```bash
pip install -r requirements.txt
```

## Optional PhysPK backend

To use the exact mechanistic simulator, provide a local licensed PhysPK export
using one of the following options:

- place it next to `src/deck_interface.py`
- place it inside `src/vendor/physpk_backend/`
- set the `PHYSPK_DECK_STEM` environment variable to the exported deck stem

The proprietary backend must not be committed to this repository.

Example in PowerShell:

```powershell
$env:PHYSPK_DECK_STEM = "C:\path\to\dkd_win64_gcc64v11_2"
```

## Running the workflow

Launch the graphical interface:

```bash
py -3.11 src/gui_launcher.py
```

Or run the command-line interface:

```bash
py -3.11 src/main.py --data_file path\to\your_data.txt --init_method prev_run --n_iter 40 --exp_name my_run
```

The exact arguments required depend on the selected analysis and the
availability of the local mechanistic backend.

## Running the tests

The public tests use synthetic or mocked data and do not require redistribution
of the proprietary PhysPK model:

```bash
py -3.11 -m pytest
```

## Curated results

The `results/` folder contains selected derived outputs, including:

- scenario-level summary tables
- final estimates for the main scenarios
- SAEM convergence traces
- emulator validation metrics
- sparse-sampling summaries
- sensitivity-analysis summaries and plots

These files were selected to support reproducibility and inspection without
exposing restricted raw data or proprietary simulator artefacts.

## Citation

If you use this software, please cite the associated article:

> Sánchez-Herrero S, García-Álvarez D, Calvet L, Juan AA.  
> Domain-validated residual XGBoost surrogates for model-informed
> biopharmaceutics and population pharmacokinetic inference.  
> *European Journal of Pharmaceutics and Biopharmaceutics* (2026).  
> https://doi.org/10.1016/j.ejpb.2026.115260

Citation metadata are also provided in `CITATION.cff`.

## License

The non-proprietary source code in this repository is released under the MIT
License. See `LICENSE` for details.

The PhysPK®/EcosimPro model, exported runtime libraries, source datasets, and
other third-party materials are not covered by this license.
