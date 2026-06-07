#!/usr/bin/env python3
"""
Cross-platform batch runner for the cross-country zero-shot experiments.

Replaces the per-method PowerShell scripts (run_experiments_*.ps1), which had
hard-coded Windows output paths (c:\\Users\\minelab\\...) and only ran on Windows.

This runner:
  - works on macOS / Linux / Windows (uses sys.executable, pathlib)
  - streams each run to the console AND saves a UTF-8 log to results/
    (the old PowerShell Tee-Object produced UTF-16, which the parsers choke on)
  - is shared across methods so TSET / DMC / AMCv2 use one entry point

Usage:
    # run TSET on all six countries (CPU on a Mac: add --device -1)
    python run_experiments.py --method tset
    python run_experiments.py --method tset --device -1

    # one country only (smoke test before the full batch)
    python run_experiments.py --method tset --countries iran --device -1

    # other pending methods
    python run_experiments.py --method dmc
    python run_experiments.py --method amcv2

Run this from the src/ directory (the experiment scripts resolve the dataset
via base_dir = cwd().parent, i.e. the project root, which must contain
data/processed/<country>/).
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ALL_COUNTRIES = ["china", "iran", "UAE", "cuba", "russia", "venezuela"]

# Per-method config: the script to call, the log-file prefix, and the
# method-specific CLI args (shared args like --gnn/--lr are added below).
METHODS = {
    "tset": {
        "script": "run_MultiModalGNN_CrossAttention_CrossCountry_TSET.py",
        "prefix": "TSET",
        "extra": ["--focal_gamma", "2.0", "--focal_alpha", "0.75",
                  "--coral_weight", "500.0", "--sotm_threshold", "0.3"],
    },
    # Ablation: TSET with Focal Loss disabled (BCE) and coral_weight back to
    # DFA's 1.0 -> isolates whether Focal Loss is what hurts the imbalanced
    # countries (e.g. cuba). Effectively "DFA + SOTM".
    "tset_bce": {
        "script": "run_MultiModalGNN_CrossAttention_CrossCountry_TSET.py",
        "prefix": "TSETbce",
        "extra": ["--loss_type", "bce", "--coral_weight", "1.0",
                  "--sotm_threshold", "0.3"],
    },
    "dmc": {
        "script": "run_MultiModalGNN_CrossAttention_CrossCountry_DMC.py",
        "prefix": "DMC",
        "extra": ["--focal_gamma", "2.0", "--focal_alpha", "0.75",
                  "--coral_weight", "500.0"],
    },
    "amcv2": {
        "script": "run_MultiModalGNN_CrossAttention_CrossCountry_AMC_v2.py",
        "prefix": "AMCv2",
        "extra": ["--loss_type", "bce", "--coral_weight", "50.0"],
    },
    # Direction 3: Channel-Specific DFA. Coverage-gated channel fusion + coverage
    # prior, BCE loss (DFA inheritance), channel-specific CORAL on coRT/coURL.
    "csdfa": {
        "script": "run_MultiModalGNN_CrossAttention_CrossCountry_CSDFA.py",
        "prefix": "CSDFA",
        "extra": ["--loss_type", "bce", "--cov_lambda", "1.0",
                  "--coral_on", "channel", "--coral_weight", "1.0",
                  "--coral_channels", "coRT,coURL"],
    },
    # Ablation: gating ON but coverage prior OFF (lambda=0) -> isolates the
    # contribution of the log-coverage prior over pure masked fusion.
    "csdfa_noprior": {
        "script": "run_MultiModalGNN_CrossAttention_CrossCountry_CSDFA.py",
        "prefix": "CSDFAnoprior",
        "extra": ["--loss_type", "bce", "--cov_lambda", "0.0",
                  "--coral_on", "channel", "--coral_weight", "1.0",
                  "--coral_channels", "coRT,coURL"],
    },
    # Ablation: CORAL disabled -> the method's gains should come from the
    # coverage-gated fusion, not alignment (text-proj CORAL is a known no-op).
    "csdfa_nocoral": {
        "script": "run_MultiModalGNN_CrossAttention_CrossCountry_CSDFA.py",
        "prefix": "CSDFAnocoral",
        "extra": ["--loss_type", "bce", "--cov_lambda", "1.0",
                  "--coral_on", "none"],
    },
    # Decisive control: gating OFF, prior OFF, CORAL OFF -> bare 5-channel
    # backbone with plain per-node attention (AMC-style). If this matches full
    # CS-DFA, none of direction 3's mechanisms contribute and the score is the
    # backbone alone.
    "csdfa_nogate": {
        "script": "run_MultiModalGNN_CrossAttention_CrossCountry_CSDFA.py",
        "prefix": "CSDFAnogate",
        "extra": ["--loss_type", "bce", "--gating", "off",
                  "--cov_lambda", "0.0", "--coral_on", "none"],
    },
}

# Shared hyper-parameters (identical across the PowerShell scripts).
SHARED_ARGS = [
    "--epochs", "1000",
    "--lr", "1e-2",
    "--early", "20",
    "--check", "1",
    "--gnn", "sage",
    "--embed_type", "positional_degree",
    "--latent", "128",
    "--splits", "5",
    "--val_metric", "f1_macro",
]


def run_country(method_cfg, country, device, src_dir, results_dir):
    log_path = results_dir / f"zero-shot_{method_cfg['prefix']}_{country}.txt"
    cmd = [
        sys.executable, "-u", method_cfg["script"],
        "--dataset", country,
        "--device", device,
        *SHARED_ARGS,
        *method_cfg["extra"],
    ]
    print("=" * 60)
    print(f"  {method_cfg['prefix']}: target = {country}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {log_path}")
    print("=" * 60, flush=True)

    # Multi-channel methods (csdfa/amc/dmc) keep all six countries' five subnet
    # edge_indices resident on the GPU, which fragments memory on smaller cards
    # (e.g. Colab T4 15GB) -> a single large message-passing alloc then fails
    # even though most reserved memory is idle. expandable_segments lets PyTorch
    # reuse that idle reserved pool, which clears the OOM in practice.
    child_env = {**os.environ}
    child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # Stream stdout/stderr to console and tee into a UTF-8 log file.
    # buffering=1 -> line-buffered so the log is monitorable in real time.
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_f:
        proc = subprocess.Popen(
            cmd, cwd=str(src_dir), env=child_env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_f.write(line)
        proc.wait()

    status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
    print(f"\n  {country}: {status} -> {log_path}\n", flush=True)
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=sorted(METHODS),
                        help="which experiment variant to run")
    parser.add_argument("--countries", nargs="+", default=ALL_COUNTRIES,
                        help="subset of target countries (default: all six)")
    parser.add_argument("--device", default="0",
                        help="device id passed to the script; use -1 to force CPU (e.g. on a Mac)")
    args = parser.parse_args()

    src_dir = Path(__file__).resolve().parent
    project_root = src_dir.parent
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    data_dir = project_root / "data" / "processed"
    if not data_dir.exists():
        print(f"[WARN] {data_dir} does not exist. The experiment scripts expect "
              f"data/processed/<country>/ with the pickle + sbert_nodeattributes_*.pt "
              f"files. Put the dataset there before running.\n", file=sys.stderr)

    method_cfg = METHODS[args.method]
    failures = []
    for country in args.countries:
        rc = run_country(method_cfg, country, args.device, src_dir, results_dir)
        if rc != 0:
            failures.append(country)

    print("=" * 60)
    if failures:
        print(f"  {method_cfg['prefix']}: done with FAILURES: {', '.join(failures)}")
        sys.exit(1)
    print(f"  {method_cfg['prefix']}: all countries complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
