# CS-DFA: Channel-Specific Decoupled Feature Alignment

Cross-country zero-shot information-operation (IO) account detection.

- Model / training script: [`src/run_MultiModalGNN_CrossAttention_CrossCountry_CSDFA.py`](../src/run_MultiModalGNN_CrossAttention_CrossCountry_CSDFA.py)
- Batch runner: [`src/run_experiments.py`](../src/run_experiments.py) (`--method csdfa` and ablation variants)
- Colab notebook: [`colab_run.ipynb`](../colab_run.ipynb) (§8)

---

## 1. Motivation

The strongest existing baseline, **DFA** (Decoupled Feature Alignment), merges the
five behavioural co-activity sub-networks of each country
(`coRT`, `coURL`, `hashSeq`, `fastRT`, `tweetSim`) into **one** graph and runs a
single GNN over it. Our per-country failure analysis
(`results/diagnostics/failure_analysis_summary.csv`) showed a problem with this:

- The five sub-networks have **wildly different coverage** per country. For example,
  for **China** only **3–7%** of accounts participate in `hashSeq`/`fastRT`/`tweetSim`,
  while `coRT`/`coURL` cover 58%/91%.
- A node that does **not** participate in a sub-network is an *isolated node* in that
  channel's graph. A GNN over that channel produces, for such a node, a pure
  pass-through of its own features with **no neighbour aggregation** — i.e. noise.
- When all five sub-networks are merged into one graph (DFA), these noisy/sparse
  channels dilute the strong, highly-transferable coordination signal carried by
  `coRT`/`coURL`.

**Hypothesis.** If we (a) keep the five channels separate and (b) let each node fuse
*only* the channels it actually belongs to, we should recover the transferable
coordination signal and improve cross-country zero-shot transfer — especially for
low-coverage countries like China.

CS-DFA combines the **DFA decoupling philosophy** with the **five-subnet channel
view** from the AMC/DMC models, and adds two coverage-aware mechanisms.

---

## 2. Method

### 2.1 Architecture

```
text + struct features
        │  (decoupled cross-attention projection, inherited from DFA)
        ▼
 multimodal node features
        │
   ┌────┴────┬─────────┬─────────┬──────────┐
 ChannelGNN ChannelGNN ChannelGNN ChannelGNN ChannelGNN     ← one GNN per sub-network
  (coRT)    (coURL)   (hashSeq)  (fastRT)  (tweetSim)
   └────┬────┴─────────┴─────────┴──────────┘
        ▼
 Coverage-Gated Fusion  ← per-node attention over the 5 channel embeddings
        ▼
   classifier → IO / non-IO
```

### 2.2 The two coverage-aware mechanisms

1. **Coverage-gated fusion (core idea).**
   For every node `n` and channel `c`, a coverage mask `m[n,c] ∈ {0,1}` records whether
   `n` actually participates in sub-network `c`. Before the per-node attention softmax,
   the score of any channel a node does **not** belong to is set to `-inf`, so the node
   fuses **only over channels it is genuinely part of**. For China this zeroes out the
   3–7%-coverage noisy channels for ~95% of nodes; for high-coverage countries it is
   nearly a no-op.

2. **Coverage prior.**
   The per-node attention scores additionally receive `λ · log(coverage_c)`, giving
   structurally-stable, high-coverage channels (`coRT`/`coURL`) a head start. `λ` is
   controlled by `--cov_lambda` (`λ=0` recovers pure gating).

### 2.3 Loss and alignment

- **Loss: BCE by default** (inherited from DFA). The earlier TSET diagnosis showed
  Focal Loss *hurts* extreme-imbalance countries (e.g. Cuba), so we do **not** use it
  by default. `--loss_type focal` is available for the loss-routing line.
- **CORAL: applied to the stable channels' embeddings** (`--coral_on channel`, default
  `coRT,coURL`). The DFA/TSET diagnosis found CORAL on the *text projection* is a no-op
  (source-pair covariance ≈ 0). The script therefore logs **both** the text-projection
  CORAL and the channel-embedding CORAL every check epoch, so the two can be compared
  directly.

### 2.4 Evaluation protocol

True **zero-shot cross-country**: the model is trained only on the five *source*
countries and evaluated on the held-out *target* country's test split. Five random
splits, `f1_macro` reported as mean ± std. Per-subnet test slices
(`TEST_coRT`, … `TEST_tweetSim`) are logged for diagnosis.

---

## 3. How to run

From `src/` with the dataset present at `<project_root>/data/processed/<country>/`:

```bash
# main method, all six countries
python run_experiments.py --method csdfa --device 0

# ablations
python run_experiments.py --method csdfa_nocoral  --device 0   # CORAL off
python run_experiments.py --method csdfa_noprior  --device 0   # coverage prior off (λ=0)
python run_experiments.py --method csdfa_nogate   --device 0   # gating off (bare backbone)

# source-data efficiency (paper Sec. "Source-Data Efficiency", Table data-eff):
# train on 10% / 50% of the labeled source-country accounts (100% = plain run)
python run_experiments.py --method csdfa --source_frac 0.1 --device 0
python run_experiments.py --method csdfa --source_frac 0.5 --device 0
python run_experiments.py --method dfa   --source_frac 0.1 --device 0
python run_experiments.py --method dfa   --source_frac 0.5 --device 0
```

`--source_frac f` subsamples the **labeled source-country training accounts** only:
per source country, a fixed class-stratified (IO vs organic) subset of `f * n`
accounts is drawn once with a seed derived from the run seed and the country
index, so the pool is identical across epochs and splits. The per-epoch batch
of 128 is then sampled from that pool. Target-country data and all evaluation
splits are untouched, and `--source_frac 1.0` (the default) is an exact no-op.
Each run logs one `[SourceFrac] <country>: organic kept/total, IO kept/total`
line per source country — diff these against a 100% run to audit the sampling.
Logs are tagged with the fraction (`zero-shot_CSDFA_frac10_<country>.txt` under
`results/csdfa_logs/`, `zero-shot_DFA_frac10_<country>.txt` under
`results/dfa_logs/`) so 10%/50% runs never overwrite the 100% logs.

On a 15 GB GPU (e.g. Colab T4), the multi-channel methods keep all six countries'
five sub-network edge indices resident on the GPU, which fragments memory. The runner
launches each country subprocess with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
to avoid the resulting OOM.

---

## 4. Results

### 4.1 Main result — CS-DFA vs DFA (zero-shot TEST `f1_macro`)

| Country    | DFA (baseline) | **CS-DFA** | Δ |
|------------|:-:|:-:|:-:|
| china      | 0.603 | **0.837** | **+0.234** |
| iran       | 0.706 | **0.998** | **+0.291** |
| UAE        | 0.880 | **0.993** | +0.113 |
| cuba       | 0.856 | **0.888** | +0.032 |
| russia     | 0.836 | **1.000** | +0.164 |
| venezuela  | 0.768 | **0.998** | +0.230 |
| **average**| **0.775** | **0.952** | **+0.177** |

CS-DFA beats DFA on **all six** countries, +0.177 on average.

### 4.2 Ablation — which component matters?

All variants use BCE; they differ only in the fusion / alignment components.

| Variant | gating | prior (λ) | CORAL | avg TEST `f1_macro` |
|---|:-:|:-:|:-:|:-:|
| **CS-DFA (full)**     | on  | 1.0 | channel | **0.952** |
| `csdfa_nocoral`       | on  | 1.0 | off     | 0.950 |
| `csdfa_noprior`       | on  | 0.0 | channel | 0.952 |
| `csdfa_nogate`        | **off** | 0.0 | off | **0.686** |
| DFA baseline          |  –  |  –  |  –      | 0.775 |

Per-country `f1_macro` for the decisive control (`csdfa_nogate`):

| | china | iran | UAE | cuba | russia | venezuela | avg |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| CS-DFA (full) | 0.837 | 0.998 | 0.993 | 0.888 | 1.000 | 0.998 | 0.952 |
| nogate        | 0.510 | 0.671 | 0.817 | 0.619 | 0.767 | 0.733 | 0.686 |
| Δ             | −0.33 | −0.33 | −0.18 | −0.27 | −0.23 | −0.27 | **−0.27** |

---

## 5. Analysis

**Coverage-gated fusion is the contribution; CORAL and the coverage prior are inert.**

- Turning **CORAL off** (`csdfa_nocoral`) changes the average by 0.002.
- Turning the **coverage prior off** (`csdfa_noprior`, λ=0) changes it by 0.000.
  → The channel-specific CORAL term and the additive coverage prior do **not** drive the
  result. (The channel-embedding CORAL is numerically non-zero, unlike the
  text-projection CORAL which is confirmed ≈ 0 — but it is dominated by the task loss
  and removing it has no effect.)
- Turning **gating off** (`csdfa_nogate`) collapses the average from **0.952 → 0.686**,
  a 0.27 drop on every country. Notably the bare five-channel backbone (0.686) is
  **worse than single-graph DFA (0.775)** — so simply *splitting* into channels is not
  enough; the coverage gate is what lifts it above DFA.

**Mechanism.** With gating off, per-node attention spreads roughly uniformly over all
five channels (e.g. China ≈ 0.21/0.25/0.16/0.18/0.19), so each node is pulled around by
its isolated-channel noise. With gating on, attention concentrates on the channels the
node truly belongs to (China ≈ coRT 0.45 / coURL 0.50, noisy channels ≈ 0.00x), which
recovers the transferable coordination signal.

**This also argues against trivial label leakage.** We audited the data path
(`data_loader.py`, `get_gnn_embeddings`, `degree_to_one_hot`): target labels never enter
training (only source countries are trained on), structural features are pure node
degree, and the splits are used only for evaluation / model selection. If the high
scores were a leak any model could exploit, `csdfa_nogate` would also be high — it is
not (0.686). The performance depends specifically on the gating mechanism.

---

## 6. Caveats

1. **Model selection on the target validation set.** Like the DFA/TSET baselines,
   early-stopping picks the checkpoint that maximises the *target* validation
   `f1_macro`. For several countries this validation score saturates at 1.0 within
   ~3 epochs, so the higher-capacity multi-channel model can be selected optimistically;
   absolute test numbers (especially russia/iran/venezuela ≈ 1.0) are likely
   **optimistic**. This does not affect the *relative* ablation conclusions (all variants
   share the protocol). A `last-epoch` or `source-val` selection number would tighten the
   absolute claims.
2. **Low-coverage sub-network slices are high-variance.** The
   `hashSeq`/`fastRT`/`tweetSim` per-subnet `f1_macro` carry ±0.20–0.25 std (few samples);
   the headline average is carried by the high-coverage `coRT`/`coURL` slices.

---

## 7. Takeaway

For cross-country zero-shot IO detection, **splitting behaviour into channels is not
enough on its own** — the key is **coverage-gated fusion**, letting each node fuse only
the behavioural channels it genuinely participates in. The ablation shows this gate is
responsible for the entire +0.27 gain over the bare multi-channel backbone, while
channel-level CORAL alignment and an additive coverage prior contribute nothing
measurable in this setting.
