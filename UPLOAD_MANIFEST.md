# Public Repository Manifest

This repository contains the curated non-proprietary components of the
surrogate-assisted SAEM workflow.

## Included

- `src/`: core Python implementation
- `scripts/`: manuscript figure-generation scripts
- `results/`: curated derived summary results
- `tests/`: synthetic and mocked tests
- `README.md`: repository documentation
- `requirements.txt`: Python dependencies
- `pytest.ini`: test configuration
- `CITATION.cff`: citation metadata
- `LICENSE`: MIT License

## Intentionally excluded

- proprietary PhysPK®/EcosimPro model source
- exported or compiled PhysPK runtime libraries
- raw source pharmacokinetic datasets
- trained emulator artefacts
- local checkpoints and bulky experiment outputs
- compiled files and caches

Users wishing to execute the exact mechanistic path must provide their own
licensed PhysPK® export locally.
