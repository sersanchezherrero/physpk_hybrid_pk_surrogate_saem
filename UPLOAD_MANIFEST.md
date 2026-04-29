# Upload Manifest

This folder is the curated public copy prepared for GitHub upload.

## Included at the repository root

- `main.py`
- `config.py`
- `deck_interface.py`
- `saem.py`
- `ml_utils.py`
- `bootstrap_utils.py`
- `diagnostics.py`
- `init_utils.py`
- `models.py`
- `nca_utils.py`
- `sensitivity_analysis.py`
- `gui_launcher.py`
- `README.md`
- `.gitignore`
- `requirements.txt`

## Included subfolders

- `tests/`
  Synthetic and mocked unit/integration tests
- `scripts/`
  Figure-generation and manuscript asset scripts
- `paper/`
  Curated LaTeX manuscript and final manuscript figures
- `results/`
  Curated derived outputs, summary tables, traces, and sensitivity artifacts
- `docs/`
  Public-release notes, data format, and proprietary component guidance
- `examples/`
  GUI launcher batch file and example commands
- `data/`
  Empty public data area plus an input template
- `vendor/physpk_backend/`
  Placeholder folder for a local non-versioned PhysPK export

## Explicitly excluded from upload

- local PhysPK runtime DLLs and generated Python wrappers
- the company-owned mechanistic model
- raw clinical datasets
- large `outputs/` experiment folders
- compiled artifacts and caches
