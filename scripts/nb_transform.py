#!/usr/bin/env python3
"""One-shot cleanup of SMM_Colab.ipynb: translate markdown to English,
drop stale/duplicate cells, reorder the data-efficiency section, and
upgrade the results parser. Operates on the Drive copy in place (backup
written alongside). Designed to run in the Colab terminal."""
import json, re, shutil, sys

PATH = '/content/drive/MyDrive/Colab Notebooks/SMM_Colab.ipynb'
BACKUP = PATH.replace('.ipynb', '.pre-cleanup-backup.ipynb')

nb = json.load(open(PATH))
cells = nb['cells']
assert len(cells) == 37, f'expected 37 cells, found {len(cells)} — re-inventory first'

def src(i):
    return ''.join(cells[i]['source'])

def expect(i, prefix):
    s = src(i)
    assert s.startswith(prefix), f'cell {i} does not start with {prefix!r}: {s[:60]!r}'

def set_md(i, text):
    assert cells[i]['cell_type'] == 'markdown', i
    cells[i]['source'] = [l + '\n' for l in text.split('\n')]
    if cells[i]['source']:
        cells[i]['source'][-1] = cells[i]['source'][-1].rstrip('\n')

def set_code(i, text):
    assert cells[i]['cell_type'] == 'code', i
    cells[i]['source'] = [l + '\n' for l in text.split('\n')]
    if cells[i]['source']:
        cells[i]['source'][-1] = cells[i]['source'][-1].rstrip('\n')

# ---- sanity anchors (original indices) ----
expect(2, '!nvidia-smi')
expect(4, 'from google.colab import drive')
expect(8, 'import os, shutil')          # old clone cell (delete)
expect(9, '%cd /content')               # current clone cell (keep)
expect(16, '%cd {RUN_DIR}/src\n!python run_experiments.py --method tset')
expect(26, 'import glob\nfs = sorted')  # scratch parse (delete)
expect(27, '%cd /content/smm/src')      # data-efficiency batch (keep, move)
expect(36, 'import glob, re')           # results parser (rewrite)

# ---- markdown translations (original indices) ----
set_md(0, """# CS-DFA — Zero-Shot Cross-Country Detection of Information Operations

**Colab GPU walkthrough** for our Social Media Mining (NCU CE7066) final project,
which extends **IOHunter** (AAAI 2025) with **CS-DFA**: channel-specific GNNs +
coverage-gated fusion for zero-shot cross-country influence-operation account
detection.

- **Repo & paper:** https://github.com/NelsonKCT/SMM_Final_Project (paper in `paper/`)
- **Data** (~3.6 GB, the six-country IOHunter release) lives on Google Drive and is
  symlinked in; **code** is cloned fresh from GitHub every session.
- **What this notebook covers:** environment setup → data sanity checks → a smoke
  test → the six-country CS-DFA main experiment → the three mechanism ablations →
  the 10%/50% source-data-efficiency runs → a final results summary.
- **Runtime:** T4 GPU (Runtime → Change runtime type → T4 GPU). One six-country
  batch takes ≈ 20–40 min.

The training cells write per-country logs to `results/` on Drive, so every number
in the paper can be traced back to a log file produced here.""")

set_md(1, """## 1. Check the GPU""")

set_md(3, """## 2. Mount Google Drive

The dataset (and the `results/` log directory) live on Drive; after authorizing,
Drive appears under `/content/drive/MyDrive/`.""")

set_md(5, """## 3. Project paths

`PROJECT_DIR` must contain `data/processed/<country>/` for the six countries
(china, iran, UAE, cuba, russia, venezuela).""")

set_md(7, """## 4. Get the latest code from GitHub

Code (git) and data (Drive) are kept separate: each session clones the latest
`main` to local disk, then symlinks `data/` and `results/` back to Drive — data
is too large to copy, and logs written to `results/` survive the session.""")

set_md(11, """## 5. Install dependencies

Colab pre-installs CUDA-enabled torch; we only add `torch_geometric` and
`mlflow`. (No compiled `torch_scatter` needed for GCN/SAGE/GAT in recent
PyG versions.)""")

set_md(13, """## 6. Smoke test — one country through the full pipeline

Runs CS-DFA with Iran as the zero-shot target (smallest data). Verifies the GPU
pipeline end-to-end in a few minutes; expect a final `[TEST] f1_macro` around
0.99 in the log tail.""")

set_md(21, """## 7. CS-DFA — Channel-Specific DFA with Coverage-Gated Fusion

The five behavioral sub-networks (`coRT`, `coURL`, `hashSeq`, `fastRT`,
`tweetSim`) cover very different fractions of accounts per country (China: three
channels at 3–7%). CS-DFA therefore:

- runs **one GNN per sub-network** instead of merging them into a single graph;
- fuses the per-channel embeddings with **coverage-gated attention** — a binary
  participation mask sets the score of channels an account does not belong to
  to −∞, so isolated-node noise never enters the fused representation;
- adds an optional coverage prior `λ·log(coverage)` and channel-level CORAL on
  the stable channels (both shown to be **inert** by the ablations below);
- keeps BCE loss (an earlier diagnosis showed Focal loss hurts extreme-imbalance
  targets like Cuba).""")

set_md(22, """### 7.1 Data sanity check — precomputed edge indices

Each country needs `edge_index_<subnet>.th` for all five sub-networks (computing
them from scratch on the first run is very slow). Expect `5/5` for all six
countries.""")

set_md(28, """### 7.2 Main experiment — all six countries

Zero-shot protocol: for each target country, train on the other five and
evaluate on the held-out target test split; 5 random splits, mean ± std
`f1_macro`. Logs go to `results/csdfa_logs/` on Drive. Expect CS-DFA to beat the
DFA baseline (avg 0.775) on every country, with the largest gains on the
low-coverage targets (China, Venezuela).

*The second cell resumes the batch for the remaining countries after the
first cell hit a T4 memory limit mid-batch (the runner now sets
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` itself, which fixed this).*""")

set_md(31, """### 7.3 Ablations — which mechanism actually matters?

Three controls, in decreasing order of importance:

1. `csdfa_nogate` — **the decisive control**: gating off (+ prior off + CORAL
   off) = a bare five-channel backbone. If coverage-gated fusion is the real
   contribution, this should collapse. *(It does: avg 0.952 → 0.686, below the
   merged-graph DFA baseline.)*
2. `csdfa_nocoral` — CORAL off. *(Changes the average by 0.002 → inert.)*
3. `csdfa_noprior` — coverage prior λ=0. *(Changes it by 0.000 → inert.)*""")

set_md(35, """## 8. Results summary — parse every log

Reads all final logs from `results/` on Drive and prints the tables the paper
uses: the six-country CS-DFA main result (overall + per-subnet slices) and the
10%/50%/100% source-data-efficiency comparison for both CS-DFA and DFA.""")

# ---- code translations (behavior-preserving) ----
set_code(6, """import os

# Point this at the Drive folder that holds data/ (code is cloned by the next
# section and does not need to live on Drive)
PROJECT_DIR = '/content/drive/MyDrive/SMM_Final_Project'

assert os.path.isdir(os.path.join(PROJECT_DIR, 'data', 'processed')), \\
    f'data/processed/ not found — upload the dataset to: {PROJECT_DIR}'
print('data path OK')
print('country folders:', sorted(os.listdir(os.path.join(PROJECT_DIR, 'data', 'processed'))))""")

# cell 9 (clone): translate comments/messages but preserve logic exactly.
c9 = src(9)
c9_new = (c9
    .replace("# 若 repo 是 private，改用帶 token 的 URL（PAT 在 GitHub Settings -> Developer settings 產生）：",
             "# For a private repo, use a token URL instead (create a PAT under")
    .replace("# REPO_URL = 'https://<GITHUB帳號>:<PAT>@github.com/NelsonKCT/SMM_Final_Project.git'",
             "# GitHub Settings -> Developer settings):\n# REPO_URL = 'https://<USER>:<PAT>@github.com/NelsonKCT/SMM_Final_Project.git'")
    .replace("# 1) 砍掉舊的，clone 最新 code 到本地", "# 1) Remove any stale copy and clone the latest code locally")
    .replace("# clone 失敗就直接停，避免後面 symlink 把問題蓋掉", "# Stop right away if the clone failed, before the symlinks hide the problem")
    .replace("'clone 失敗，先看上面 git 的輸出'", "'clone failed — check the git output above'")
    .replace("# 2) data 太大不複製，symlink 指回 Drive（只讀）", "# 2) data is too large to copy: symlink it back to Drive (read-only)")
    .replace("# 3) results 也 symlink 回 Drive，讓 log 直接保留", "# 3) results is symlinked to Drive too, so logs persist across sessions")
    .replace("# 3) results symlink 回 Drive，斷線也能保留已跑完的 log", "# 3) results is symlinked to Drive too, so finished logs survive disconnects")
    .replace("# 4) 驗證：抓到的 commit + 關鍵檔案/資料夾是否存在", "# 4) Verify: the checked-out commit + key files/folders exist")
    .replace("# 4) 驗證", "# 4) Verify")
    .replace("--- 驗證 ---", "--- verify ---")
    .replace("print('run_experiments 存在：'", "print('run_experiments.py exists:'")
    .replace("print('CSDFA 檔存在：'", "print('CSDFA script exists:'")
    .replace("nogate 檔有 --gating: ", "nogate script has --gating: ")
    .replace("nogate 檔有 --gating：", "nogate script has --gating:")
)
cells[9]['source'] = [l + '\n' for l in c9_new.split('\n')]
cells[9]['source'][-1] = cells[9]['source'][-1].rstrip('\n')

c12 = src(12)
c12_new = (c12
    .replace("# 只裝 torch_geometric 與 mlflow。", "# Only torch_geometric and mlflow are needed.")
    .replace("# 不要裝 node2vec：它會拉進 gensim 把 numpy 降到 <2，與 Colab 的 numpy 2.x", "# Do NOT install node2vec: it pulls in gensim, downgrades numpy below 2,")
    .replace("# 套件 binary 不相容（numpy.dtype size changed）。TSET/DMC/AMCv2 不需要 node2vec，", "# and breaks binary compatibility with Colab's numpy 2.x. Nothing here needs")
    .replace("# my_utils.py 已把該 import 改成 lazy（只有跑 Node2Vec baseline 才會用到）。", "# it: my_utils.py imports it lazily (only the Node2Vec baseline uses it).")
    .replace("# networkx / scikit-learn / scipy / pandas Colab 已有，不用裝", "# networkx / scikit-learn / scipy / pandas ship with Colab already")
    .replace("'— 依賴 OK'", "'— dependencies OK'")
)
cells[12]['source'] = [l + '\n' for l in c12_new.split('\n')]
cells[12]['source'][-1] = cells[12]['source'][-1].rstrip('\n')

# ---- new results parser (cell 36) ----
set_code(36, """import glob, re

RESULTS = f'{PROJECT_DIR}/results'

def grab_f1(path):
    txt = open(path, encoding='utf-8', errors='replace').read()
    m = re.search(r'\\[TEST\\] f1_macro:\\s*([\\d.]+)(?:\\+-([\\d.]+))?', txt)
    return (float(m.group(1)), float(m.group(2) or 0)) if m else None

COUNTRIES = ['china', 'iran', 'UAE', 'cuba', 'russia', 'venezuela']

def table(title, pattern):
    rows = {}
    for f in glob.glob(pattern, recursive=True):
        c = re.search(r'_(\\w+)\\.txt$', f).group(1)
        rows[c] = grab_f1(f)
    if not rows:
        return
    print(f'\\n{title}')
    vals = []
    for c in COUNTRIES:
        if c in rows and rows[c]:
            m, s = rows[c]
            vals.append(m)
            print(f'  {c:10s} {m:.3f} +- {s:.3f}')
    if vals:
        print(f'  {"AVERAGE":10s} {sum(vals)/len(vals):.3f}   (n={len(vals)})')

table('CS-DFA main (100% source data)', f'{RESULTS}/**/zero-shot_CSDFA_[!f]*.txt')
table('CS-DFA ablation: no CORAL',      f'{RESULTS}/**/zero-shot_CSDFAnocoral_*.txt')
table('CS-DFA ablation: no prior',      f'{RESULTS}/**/zero-shot_CSDFAnoprior_*.txt')
table('CS-DFA ablation: no gate',       f'{RESULTS}/**/zero-shot_CSDFAnogate_*.txt')
table('CS-DFA @ 10% source data',       f'{RESULTS}/csdfa_logs/zero-shot_CSDFA_frac10_*.txt')
table('CS-DFA @ 50% source data',       f'{RESULTS}/csdfa_logs/zero-shot_CSDFA_frac50_*.txt')
table('DFA    @ 10% source data',       f'{RESULTS}/dfa_logs/zero-shot_DFA_frac10_*.txt')
table('DFA    @ 50% source data',       f'{RESULTS}/dfa_logs/zero-shot_DFA_frac50_*.txt')""")

# ---- new markdown for the data-efficiency section (inserted, then reordered) ----
data_eff_md = {
    'cell_type': 'markdown',
    'metadata': {},
    'source': [l + '\n' for l in """### 7.4 Source-data efficiency — 10% / 50% of the labeled source accounts

How much labeled source data does CS-DFA actually need? `--source_frac f` keeps a
fixed, class-stratified fraction `f` of each source country's labeled accounts
(target data and evaluation splits are untouched; `[SourceFrac]` audit lines in
the log show exactly what was kept). We rerun CS-DFA and the DFA baseline at 10%
and 50% and compare with the 100% runs above. Expect CS-DFA to stay close to its
full-data performance even at 10%.""".split('\n')],
}
data_eff_md['source'][-1] = data_eff_md['source'][-1].rstrip('\n')

# ---- reorder + delete ----
DELETE = {8, 10, 15, 16, 17, 18, 19, 20, 24, 25, 26}
keep = [c for i, c in enumerate(cells) if i not in DELETE and i != 27]
dataeff_cell = cells[27]
# find the position of the ablation %env cell (original 34) in `keep`, insert after it
anchor_src = ''.join(cells[34]['source'])
pos = next(i for i, c in enumerate(keep) if ''.join(c['source']) == anchor_src)
keep[pos + 1:pos + 1] = [data_eff_md, dataeff_cell]
nb['cells'] = keep

shutil.copyfile(PATH, BACKUP)
json.dump(nb, open(PATH, 'w'), ensure_ascii=False)
print('backup:', BACKUP)
print('final cell count:', len(keep))
for i, c in enumerate(keep):
    s = ''.join(c['source']).replace('\n', ' ')[:46]
    print(i, c['cell_type'][:2], repr(s))

# safety net: report any Chinese characters left in cell sources
import unicodedata
bad = [(i, l.strip()[:60]) for i, c in enumerate(keep) for l in c['source']
       if any('一' <= ch <= '鿿' for ch in l)]
print('REMAINING ZH LINES:', len(bad))
for i, l in bad[:10]:
    print('  zh @', i, l)
