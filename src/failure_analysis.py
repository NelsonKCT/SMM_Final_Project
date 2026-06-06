#!/usr/bin/env python3
"""
Cross-country failure-analysis diagnostics.

Answers "why are some target countries (China / Cuba / Venezuela) hard?" with
per-country structural / label statistics and a linguistic-distance matrix,
WITHOUT training anything. Read-only; safe to run while experiments are going.

Outputs (to results/):
  - failure_analysis_summary.csv : per-country nodes/edges, IO prevalence
    (label imbalance), homophily, and per-subnet node coverage
  - linguistic_mmd_matrix.csv    : 6x6 SBERT MMD matrix (text-distribution
    distance), the basis for TSET's Structure-Only Transfer Mode decision

Run from the src/ directory (resolves data via ../data/processed/).

    python failure_analysis.py
"""
import pathlib
import pickle

import numpy as np
import pandas as pd
import torch
from torch_geometric.utils import homophily

from my_utils import get_edge_index_from_networkx

ALL_COUNTRIES = ['china', 'iran', 'UAE', 'cuba', 'russia', 'venezuela']
SUBNETS = ['coRT', 'coURL', 'hashSeq', 'fastRT', 'tweetSim']
FILTER_TH = 0.7


def compute_mmd(x, y, kernel_bandwidths=(0.1, 0.5, 1.0, 2.0, 5.0), max_n=500, seed=0):
    """MMD^2 between x and y (RBF, multiple bandwidths). Same formulation as
    the SOTM detector in run_..._TSET.py, extracted here for the full matrix."""
    g = torch.Generator().manual_seed(seed)
    if x.size(0) > max_n:
        x = x[torch.randperm(x.size(0), generator=g)[:max_n]]
    if y.size(0) > max_n:
        y = y[torch.randperm(y.size(0), generator=g)[:max_n]]

    xx, yy, xy = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    x_sq = (x ** 2).sum(1, keepdim=True)
    y_sq = (y ** 2).sum(1, keepdim=True)
    d_xx = x_sq + x_sq.t() - 2 * xx
    d_yy = y_sq + y_sq.t() - 2 * yy
    d_xy = x_sq + y_sq.t() - 2 * xy

    mmd2 = 0.0
    for bw in kernel_bandwidths:
        mmd2 += (torch.exp(-d_xx / (2 * bw ** 2)).mean()
                 + torch.exp(-d_yy / (2 * bw ** 2)).mean()
                 - 2 * torch.exp(-d_xy / (2 * bw ** 2)).mean())
    return (mmd2 / len(kernel_bandwidths)).item()


def per_country_stats(data_dir, country):
    with open(data_dir / country / f'{FILTER_TH}_datasets.pkl', 'rb') as f:
        datasets = pickle.load(f)

    graph = datasets['graph']
    labels = np.asarray(datasets['labels'])
    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()
    n_io = int(labels.sum())
    prevalence = 100.0 * n_io / len(labels)

    # Homophily on the FULL fused graph: node ids are consecutive 0..N-1 so
    # edge_index aligns with labels (no re-index mismatch from dropping nodes).
    edge_index = get_edge_index_from_networkx(graph)
    node_labels = torch.tensor(labels).long()
    if edge_index.size(1) > 0:
        node_h = homophily(edge_index, node_labels, method='node')
        edge_h = homophily(edge_index, node_labels, method='edge')
        ei_h = homophily(edge_index, node_labels, method='edge_insensitive')
    else:
        node_h = edge_h = ei_h = float('nan')

    row = {
        'country': country,
        'nodes': n_nodes,
        'edges': n_edges,
        'io_drivers': n_io,
        'io_prevalence_%': round(prevalence, 2),
        'imbalance_ratio': round((len(labels) - n_io) / max(n_io, 1), 2),
        'node_homophily': round(float(node_h), 4),
        'edge_homophily': round(float(edge_h), 4),
        'edge_insensitive_homophily': round(float(ei_h), 4),
    }
    # Per-subnet node coverage (fraction of accounts that appear in the subnet)
    for sub in SUBNETS:
        cov = 100.0 * datasets[sub].number_of_nodes() / n_nodes
        row[f'{sub}_cov_%'] = round(cov, 2)

    del datasets, graph
    return row


def load_text_sample(data_dir, country, max_n=500, seed=0):
    feats = torch.load(data_dir / country / 'sbert_nodeattributes_mostPop5.pt',
                       map_location='cpu').float()
    g = torch.Generator().manual_seed(seed)
    if feats.size(0) > max_n:
        feats = feats[torch.randperm(feats.size(0), generator=g)[:max_n]]
    return feats.contiguous()


def main():
    src_dir = pathlib.Path(__file__).resolve().parent
    data_dir = src_dir.parent / 'data' / 'processed'
    results_dir = src_dir.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    print('=== Per-country structural / label statistics ===')
    rows = []
    for c in ALL_COUNTRIES:
        print(f'  [{c}] loading...', flush=True)
        rows.append(per_country_stats(data_dir, c))
    stats_df = pd.DataFrame(rows).set_index('country')
    stats_path = results_dir / 'failure_analysis_summary.csv'
    stats_df.to_csv(stats_path)
    print('\n' + stats_df.to_string())
    print(f'\nsaved -> {stats_path}\n')

    print('=== Linguistic distance (SBERT MMD) matrix ===')
    samples = {c: load_text_sample(data_dir, c) for c in ALL_COUNTRIES}
    mmd = pd.DataFrame(index=ALL_COUNTRIES, columns=ALL_COUNTRIES, dtype=float)
    for i, ci in enumerate(ALL_COUNTRIES):
        for j, cj in enumerate(ALL_COUNTRIES):
            if i <= j:
                val = compute_mmd(samples[ci], samples[cj])
                mmd.loc[ci, cj] = val
                mmd.loc[cj, ci] = val
    mmd_path = results_dir / 'linguistic_mmd_matrix.csv'
    mmd.round(5).to_csv(mmd_path)
    print('\n' + mmd.round(5).to_string())

    # For each country: nearest source (excluding self) -> SOTM intuition
    print('\n--- Each country vs its nearest *other* country (linguistic) ---')
    for c in ALL_COUNTRIES:
        others = mmd.loc[c].drop(c)
        nearest = others.idxmin()
        print(f'  {c:11s} nearest={nearest:11s} min_MMD={others.min():.5f} '
              f'mean_to_others={others.mean():.5f}')
    print(f'\nsaved -> {mmd_path}')


if __name__ == '__main__':
    main()
