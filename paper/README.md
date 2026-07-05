# CS-DFA Conference Paper

Overleaf-ready LaTeX source turning the CE7066 poster into a short conference
paper. Template: **IEEEtran (conference)** — builds on Overleaf with zero
extra files; swap the class later if the target venue differs.

## How to build

- **Overleaf**: upload this whole `paper/` folder and compile `main.tex`
  with pdfLaTeX.
- **Locally**: `tectonic main.tex` inside `paper/`
  (`brew install tectonic`); this is how the current `main.pdf` was built.

## What is already filled in

- Full paper structure: abstract, intro (FIMI / coordination detection / GNN /
  IOHunter related work), method with equations, experiments, limitations,
  conclusion.
- All numbers currently in the tables are **real**, taken from:
  - `results/zero-shot_*` logs (CORAL / DANN / DFA per-country F1 ± std)
  - `results/diagnostics/failure_analysis_summary.csv` (dataset table)
  - `docs/CS-DFA.md` (CS-DFA main results + ablations + nogate control)
  - IOHunter paper-reported baseline (from `results/parse_results.py`)
- Hyperparameter table (Table II) copied from the CSDFA runner defaults.
- Figure 1 rendered from `figures/architecture.mmd` via mermaid-cli
  (`figures/architecture.pdf`, vector; re-render with
  `npx -y @mermaid-js/mermaid-cli -i architecture.mmd -o architecture.pdf
  --pdfFit` after editing the .mmd).
- **All 23 bibliography entries verified against official sources on
  2026-07-05** (authors / venue / pages / DOIs recorded in the `.bib`).
  Notable fixes: IOHunter got its AAAI-25 volume/pages/DOI; the GFM survey
  is now the published TPAMI version; GraphSAGE and Sun et al. 2016 got
  page numbers. Open judgment calls are documented in the header of
  `references.bib` (Rex vs. Zhitao Ying; Focal-loss page numbering;
  ICLR entries have no DOI by venue norm).

## Remaining TODOs (search `\todo{` in main.tex — renders as red text)

1. **Author emails** (names/order already follow the poster) + decide
   whether the advisor joins as co-author.

Everything else is filled in with real numbers: CS-DFA ± std (Table III,
from the Colab run logs), the 10%/50% data-efficiency runs (Table VI,
`--source_frac` on `main`), and the Limitations section was rewritten so it
does not depend on an extra model-selection experiment.

## Before submission

- Decide the venue and restyle: the content targets a short/workshop paper.
  Natural fits: **ICWSM (short/poster), ASONAM, WebSci, MISDOOM, workshops at
  WWW/KDD on integrity & misinformation**. IEEEtran already matches ASONAM.
- Check the venue's **AI-usage disclosure** policy — a template statement is
  commented out at the end of `main.tex`.
- The Limitations section deliberately discloses the target-validation
  model-selection caveat — keep it; it is the honest version of the story
  and reviewers will ask otherwise.
