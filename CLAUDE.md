# SMM_Final_Project — InfoOps Detection via GFM

Social Media Mining final project extending the AAAI 2025 paper "InfoOpsGFM"
(zero-shot influence-operation detection) with cross-country domain-adaptation
methods: DFA-GFM, CS-DFA, AMC-GFM (v1/v2), CORAL, DANN, DMC, multi-level opt.

## Stack

Python, PyTorch + PyTorch Geometric, pandas/scikit-learn/networkx/mlflow.
No `requirements.txt` / `pyproject.toml` yet — dependencies are inferred from
imports in `src/*.py`. Ask before adding a manifest; don't add one silently.

## Commands

- Setup: not yet defined (no venv/manifest committed)
- Run: PowerShell batch runners in `src/`, e.g.
  `powershell -ExecutionPolicy Bypass -File src/run_experiments_dfa.ps1`
  (per-country zero-shot runs; see `README.md` for the full list)
- Test: no test suite exists

## Layout

- `src/` — training scripts (`run_*.py`) and `.ps1` batch runners per method
- `results/` — experiment logs, plots, and `parse_*.py` result parsers
- `docs/` — method write-ups (e.g. `CS-DFA.md`)
- `paper/` — conference-paper LaTeX source (IEEEtran, Overleaf-ready); see
  `paper/README.md` for remaining TODOs before submission
- `poster_overleaf/` — poster source archive (`SMM.zip`, baposter)
- `colab_run.ipynb` — Colab entry point
- Dataset lives outside the repo (Zenodo / lab Windows machine); `data/` is gitignored

## Conventions

- Shared rules in `.claude/rules/` are **copies** of AI-Harness shared rules, not
  symlinks — git history shows commits from multiple authors (Jieyu, others)
  beyond the current GitHub admin, so absolute symlinks would break for them.
  Re-run `/adopt-project` to refresh if the harness rules change upstream.
- Project-specific rules: add new files in `.claude/rules/`, one topic per file,
  <100 lines each.
- When Claude repeats a mistake, capture it as one concrete, verifiable line
  here or in a rule file (e.g. "run `npm test` before committing", not "test
  your changes").
- Dispatch & judgment rules are global — `~/.claude/rules/model-dispatch.md` and
  `~/Documents/AI-Harness/shared/playbooks/`. If the dispatch rule is missing from
  context, read it before any bulk reading or delegation.
