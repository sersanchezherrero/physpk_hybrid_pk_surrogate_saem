# Hybrid Mechanistic-Emulator PK Workflow

Public GitHub-ready copy of the Python workflow used for surrogate-assisted
population pharmacokinetic inference with SAEM, residual XGBoost emulation,
internal bootstrap validation, sparse-sampling analyses, and manuscript figure
generation.

## What this repository contains

- Core Python source code for the analysis pipeline
- Unit and integration tests based on synthetic or mocked data
- Figure-generation scripts used for the manuscript
- A clean LaTeX paper folder with curated manuscript figures
- A curated results folder with reviewer-friendly derived outputs
- Documentation describing the expected data format and the proprietary
  components that are intentionally excluded

## What is intentionally NOT included

This public copy does **not** include:

- the proprietary PhysPK / EcosimPro model source
- generated PhysPK runtime libraries (`.dll`, `.a`, `.pyc`, exported wrappers)
- raw clinical datasets or any data with uncertain redistribution rights
- local experiment outputs, checkpoints, trained emulator artifacts, or bulky
  working folders

That means the exact mechanistic backend is optional in the public repo. The
Python workflow is preserved, but users must provide their own local PhysPK
export if they want to run the exact simulator path.

## Repository layout

```text
github_repo_upload_clean/
  main.py
  config.py
  deck_interface.py
  saem.py
  ml_utils.py
  bootstrap_utils.py
  diagnostics.py
  init_utils.py
  models.py
  nca_utils.py
  sensitivity_analysis.py
  gui_launcher.py
  requirements.txt
  .gitignore
  UPLOAD_MANIFEST.md
  docs/
  examples/
  paper/
  results/
  scripts/
  tests/
  data/
  vendor/
```

## Quick start

1. Create a clean Python environment.
2. Install the Python dependencies:

```bash
pip install -r requirements.txt
```

3. If you want to use the exact mechanistic backend, place your local exported
   PhysPK runtime next to `deck_interface.py` or inside
   `vendor/physpk_backend/`.

4. Launch the GUI:

```bash
py -3.11 gui_launcher.py
```

5. Or run the CLI directly:

```bash
py -3.11 main.py --data_file path\\to\\your_data.txt --init_method prev_run --n_iter 40 --exp_name my_run
```

## Manuscript material

The curated paper files are stored in:

- `paper/main.tex`
- `paper/ref.bib`
- `paper/figures/`

This keeps the manuscript assets versioned without mixing them with local
working outputs.

## Curated results package

The repository also includes a curated `results/` folder containing:

- scenario-level summary tables
- final estimates for the main scenarios
- SAEM convergence traces
- emulator validation metrics
- sparse-sampling summaries
- sensitivity-analysis summaries and raw sensitivity plots

These files were selected to improve reproducibility and reviewer inspection
without exposing restricted raw data or bulky local training artifacts.

## Recommended release workflow

Before making the GitHub repository public:

1. Review `docs/PROPRIETARY_COMPONENTS.md`
2. Review `docs/UPLOAD_CHECKLIST.md`
3. Confirm that no raw clinical or company-restricted files were copied
4. Confirm that `vendor/physpk_backend/` is still empty except for the README

## Notes

- The GUI and CLI remain useful even without the proprietary backend for
  command generation, synthetic tests, and manuscript asset preparation.
- The exact PhysPK execution path requires local non-redistributed artifacts.
- You may want to add a license and a `CITATION.cff` file before publishing the
  repository.
