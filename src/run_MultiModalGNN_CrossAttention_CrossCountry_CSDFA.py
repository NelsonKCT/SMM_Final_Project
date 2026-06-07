"""
CS-DFA-GFM: Channel-Specific Decoupled Feature Alignment.

Direction 3 of the SMM project. Combines the strongest baseline (DFA) with the
five-subnet channel view from AMC/DMC, but treats channels *differently* based
on how reliable each behavioural subnet is for the country at hand.

Two mechanisms, both grounded in the per-country failure analysis
(results/failure_analysis_summary.csv):

  1. Coverage-gated masked fusion (the core contribution).
     Each ChannelGNN runs on its own subnet edge_index. A node that does NOT
     participate in a subnet is an isolated node in that channel's graph -> its
     channel embedding is pure noise (just a transform of its own features, no
     neighbour aggregation). We mask those channels out per-node before the
     attention fusion, so a node only fuses over channels it actually belongs to.
     For china (hashSeq/fastRT/tweetSim coverage only 3-7%) this kills the noisy
     channels for ~95% of nodes; for cuba (tweetSim coverage 99.96%) it is
     almost a no-op -> exactly the "imbalance compensated by coverage" story,
     now mechanised.

  2. Coverage prior on the fusion weights.
     We add lambda * log(coverage_c) to the per-node attention scores, giving
     high-coverage / structurally-stable channels (coRT, coURL) a head start.
     lambda=0 recovers pure gating; large lambda -> trust coRT/coURL almost
     exclusively. This injects the failure-analysis finding as inductive bias.

Loss: BCE by default (inherited from DFA). Focal is available via --loss_type
for the imbalance-aware loss-routing line, since the TSET diagnosis showed Focal
*hurts* extreme-imbalance countries (cuba). Coverage gating (channel noise) and
loss choice (label imbalance) are orthogonal failure-mode levers and stack.

CORAL: the TSET/DFA diagnosis found CORAL on the *text projection* is a no-op
(source-pair covariance difference ~= 0 at any weight). This script therefore
applies CORAL to the *channel embeddings* of the stable channels only, and ALSO
logs the text-projection CORAL each check epoch so the two can be compared
directly -- i.e. it doubles as the "is channel-z CORAL actually non-trivial?"
diagnostic. Use --coral_on {channel,text,none} to ablate.
"""
import argparse
import os
import copy
import pickle
import random
import mlflow
import torch
from torch_geometric.nn import GCNConv, SAGEConv
import numpy as np
import pandas as pd
from tqdm import tqdm

from models import GNN
from my_utils import set_seed, setup_env, move_data_to_device, update_best_model_snapshot \
    , save_metrics, get_edge_index, handle_isolated_nodes, get_gnn_embeddings
from data_loader import create_data_loader
from model_eval import TrainLogMetrics, TestLogMetrics, eval_pred
from plot_utils import plot_losses

DEFAULT_HYPERPARAMETERS = {'train_perc': .6,
                           'val_perc': .2,
                           'test_perc': .2,
                           'num_splits': 5,
                           'aggr_type': 'mean'}
DEFAULT_TRAIN_HYPERPARAMETERS = {'input_embed': 'positional', 'epochs': 1000, 'learning_rate': 1e-3,
                                 'early_stopping_limit': 10, 'check_loss_freq': 5}
DEFAULT_MODEL_HYPERPARAMETERS = {'gnn_type': 'sage', 'latent_dim': 128, 'dropout': 0.2}
ALL_COUNTRIES = ['china', 'iran', 'UAE', 'cuba', 'russia', 'venezuela']

# Canonical channel order. Coverage masks, attention weights and logging all
# follow this order. Stable channels (used for channel-specific CORAL) are the
# first two; the high-noise channels are the last three.
SUBNETS = ['coRT', 'coURL', 'hashSeq', 'fastRT', 'tweetSim']


# =============================================================================
# Focal Loss (only used when --loss_type focal; default is BCE)
# =============================================================================
class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # weight for the positive (IO) class
        self.reduction = reduction

    def forward(self, preds, targets):
        preds = preds.clamp(1e-7, 1 - 1e-7)
        bce = -(targets * torch.log(preds) + (1 - targets) * torch.log(1 - preds))
        p_t = preds * targets + (1 - preds) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_weight * focal_weight * bce
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss


# =============================================================================
# ChannelGNN (GNN processing for a single behavioural subnet)
# =============================================================================
class ChannelGNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, out_dim, gnn_type='sage', dropout_p=0.2):
        super().__init__()
        if gnn_type == 'gcn':
            conv_block = GCNConv
        elif gnn_type == 'sage':
            conv_block = SAGEConv
        else:
            raise Exception(f"GNN type '{gnn_type}' not supported in ChannelGNN")
        self.conv1 = conv_block(num_node_features, hidden_dim)
        self.conv2 = conv_block(hidden_dim, out_dim)
        self.activation_fn = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(dropout_p)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        return x


# =============================================================================
# CoverageGatedFusion (per-node attention over channels, gated by coverage)
# =============================================================================
class CoverageGatedFusion(torch.nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn_proj = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, z_list, coverage_mask, log_prior=None, prior_lambda=0.0, gating=True):
        """
        z_list:        list of 5 tensors, each [N, hidden_dim]
        coverage_mask: [N, 5] float in {0,1}; 1 = node participates in channel c
        log_prior:     [5] log(coverage fraction) per channel, or None
        prior_lambda:  scalar weight on the coverage prior
        gating:        if False, plain per-node attention over ALL channels (no
                       coverage mask, no prior) -> the AMC-style backbone control.
        Returns fused [N, hidden_dim] and weights [N, 5].
        """
        z_stack = torch.stack(z_list, dim=1)          # [N, 5, H]
        scores = self.attn_proj(z_stack).squeeze(-1)  # [N, 5]

        if gating:
            if log_prior is not None and prior_lambda != 0:
                scores = scores + prior_lambda * log_prior.unsqueeze(0)  # broadcast [1,5]
            # Gate: a node cannot attend to a channel it does not belong to.
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(coverage_mask < 0.5, neg_inf)
            # A node covered by zero channels (shouldn't happen, the union of subnets
            # is the full graph) ends up with all-equal scores -> softmax = uniform,
            # which is a safe fallback (no NaNs).
        weights = torch.softmax(scores, dim=1)         # [N, 5]

        z_fused = torch.sum(weights.unsqueeze(-1) * z_stack, dim=1)  # [N, H]
        return z_fused, weights


# =============================================================================
# GNN_CrossAttention_CSDFA
# =============================================================================
class GNN_CrossAttention_CSDFA(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim, num_classes, num_textual_features,
                 num_structural_features, activation_fn=torch.nn.ReLU(), dropout_p=0.2,
                 gnn_type='sage'):
        super().__init__()
        self.cross_attention_to_text = torch.nn.Linear(num_structural_features, hidden_dim)
        self.cross_attention_to_struct = torch.nn.Linear(num_textual_features, hidden_dim)
        self.struct_projector = torch.nn.Sequential(
            torch.nn.Linear(num_structural_features, hidden_dim),
            torch.nn.ReLU()
        )
        self.text_projector = torch.nn.Sequential(
            torch.nn.Linear(num_textual_features, hidden_dim),
            torch.nn.ReLU()
        )
        self.joint_projector = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim * 2),
            torch.nn.ReLU()
        )

        # 5 dedicated channels for each behavioural subnet
        self.gnn_coRT = ChannelGNN(hidden_dim * 2, hidden_dim, hidden_dim, gnn_type, dropout_p)
        self.gnn_coURL = ChannelGNN(hidden_dim * 2, hidden_dim, hidden_dim, gnn_type, dropout_p)
        self.gnn_hashSeq = ChannelGNN(hidden_dim * 2, hidden_dim, hidden_dim, gnn_type, dropout_p)
        self.gnn_fastRT = ChannelGNN(hidden_dim * 2, hidden_dim, hidden_dim, gnn_type, dropout_p)
        self.gnn_tweetSim = ChannelGNN(hidden_dim * 2, hidden_dim, hidden_dim, gnn_type, dropout_p)

        self.fusion = CoverageGatedFusion(hidden_dim)

        self.classifier = torch.nn.Linear(hidden_dim, 1)
        self.output_fn = torch.nn.LogSigmoid()

    def get_text_projection(self, text_node_features, struct_node_features):
        return self.text_projector(text_node_features) * self.cross_attention_to_text(struct_node_features)

    def forward(self, text_node_features, struct_node_features, edge_indices, coverage_mask,
                log_prior=None, prior_lambda=0.0, gating=True,
                return_text_features=False, return_attention=False, return_channels=False):
        struct_projection = (
            self.struct_projector(struct_node_features)
            * self.cross_attention_to_struct(text_node_features)
        )
        text_projection = (
            self.text_projector(text_node_features)
            * self.cross_attention_to_text(struct_node_features)
        )

        multimodal_node_features = self.joint_projector(
            torch.concat([struct_projection, text_projection], dim=-1)
        )

        z_coRT = self.gnn_coRT(multimodal_node_features, edge_indices['coRT'])
        z_coURL = self.gnn_coURL(multimodal_node_features, edge_indices['coURL'])
        z_hashSeq = self.gnn_hashSeq(multimodal_node_features, edge_indices['hashSeq'])
        z_fastRT = self.gnn_fastRT(multimodal_node_features, edge_indices['fastRT'])
        z_tweetSim = self.gnn_tweetSim(multimodal_node_features, edge_indices['tweetSim'])
        z_list = [z_coRT, z_coURL, z_hashSeq, z_fastRT, z_tweetSim]

        fused, weights = self.fusion(z_list, coverage_mask, log_prior, prior_lambda, gating)

        out = torch.exp(self.output_fn(self.classifier(fused)))

        res = [out]
        if return_text_features:
            res.append(text_projection)
        if return_attention:
            res.append(weights)
        if return_channels:
            res.append(z_list)
        if len(res) == 1:
            return out
        return tuple(res)


# =============================================================================
# Multi-channel data loader (+ per-node coverage mask aligned to node order)
# =============================================================================
def build_coverage_mask(datasets, num_nodes, device):
    """[N, 5] float coverage mask in canonical SUBNETS order. Node ids are the
    consecutive 0..N-1 graph ids, identical to the row order of node_features /
    struct_node_features / the fusion output."""
    cov = np.zeros((num_nodes, len(SUBNETS)), dtype=np.float32)
    for ci, subnet in enumerate(SUBNETS):
        members = list(datasets[subnet].nodes())
        if len(members) > 0:
            cov[members, ci] = 1.0
    return torch.from_numpy(cov).to(device)


def read_all_data_csdfa(device_id, dataset_name, hyper_params, train_hyperparams, model_hyperparams):
    device, base_dir, interim_data_dir, data_dir = setup_env(device_id, dataset_name, hyper_params)
    print(data_dir)
    datasets = create_data_loader(data_dir, hyper_params['tsim_th'],
                                  hyper_params['train_perc'], hyper_params['undersampling'])
    datasets = move_data_to_device(datasets, device)

    # Global node set -> stable, consecutive 0..N-1 mapping shared by all subnets.
    global_graph = datasets['graph'].copy()
    _, global_network = handle_isolated_nodes(global_graph)
    global_nodes = list(global_network.nodes())
    num_nodes = global_network.number_of_nodes()

    subnets = SUBNETS
    edge_indices = {}
    for subnet_name in subnets:
        fname = f'edge_index_{subnet_name}.th'
        if (data_dir / fname).exists():
            print(f"  Loading pre-computed edge index for subnet '{subnet_name}' from disk...")
            edge_idx = torch.load(data_dir / fname, map_location=device)
        else:
            print(f"  Computing edge index for subnet '{subnet_name}'...")
            subnet_graph = datasets[subnet_name].copy()
            subnet_graph.add_nodes_from(global_nodes)
            _, subnet_network = handle_isolated_nodes(subnet_graph)
            edge_idx = get_edge_index(subnet_network, data_dir, type=f'_{subnet_name}')
        edge_indices[subnet_name] = edge_idx.to(device)

    # Per-node coverage mask (built from the ORIGINAL subnet membership, before
    # the isolated-node reconnection -- we want true membership, not the random
    # edges handle_isolated_nodes adds).
    coverage_mask = build_coverage_mask(datasets, num_nodes, device)

    num_mostPop = hyper_params['most_pop']
    if (data_dir / f'sbert_nodeattributes_mostPop{num_mostPop}.pt').exists():
        node_features = torch.load(data_dir / f'sbert_nodeattributes_mostPop{num_mostPop}.pt', map_location=device)
    else:
        path = str(data_dir / f'sbert_nodeattributes_mostPop{num_mostPop}.pt')
        raise Exception(f'path {path} does not exist')
    node_features = node_features.to(device)

    struct_node_features = get_gnn_embeddings(data_dir, {'type': train_hyperparams['input_embed'],
                                                          'trace_type': hyper_params['trace_type'],
                                                          'latent_dim': model_hyperparams['latent_dim'],
                                                          'seed': hyper_params['seed'],
                                                          'num_nodes': global_network.number_of_nodes(),
                                                          'graph': global_network, 'device': device,
                                                          'dataset_name': dataset_name, 'base_dir': base_dir,
                                                          'num_cores': 8,
                                                          'aggr_type': hyper_params['aggr_type']})
    struct_node_features = struct_node_features.to(device)

    return (device, base_dir, interim_data_dir, data_dir, datasets, edge_indices,
            global_network, node_features, struct_node_features, coverage_mask)


def coverage_log_prior(coverage_mask, eps=1e-3):
    """[5] log(coverage fraction) per channel, used as the fusion prior."""
    frac = coverage_mask.mean(dim=0)            # [5]
    return torch.log(frac + eps)


def create_model(model_hyperparams):
    return GNN_CrossAttention_CSDFA(
        num_node_features=model_hyperparams['latent_dim'],
        hidden_dim=model_hyperparams['latent_dim'],
        num_classes=2,
        dropout_p=model_hyperparams['dropout'],
        gnn_type=model_hyperparams['gnn_type'],
        num_textual_features=model_hyperparams['num_textual_features'],
        num_structural_features=model_hyperparams['num_structural_features']
    )


def coral_loss(source, target):
    d = source.size(1)
    ns = source.size(0)
    nt = target.size(0)
    source_mean = torch.mean(source, dim=0, keepdim=True)
    source_center = source - source_mean
    xm = torch.matmul(source_center.t(), source_center) / (ns - 1)
    target_mean = torch.mean(target, dim=0, keepdim=True)
    target_center = target - target_mean
    yt = torch.matmul(target_center.t(), target_center) / (nt - 1)
    loss = torch.sum((xm - yt) ** 2) / (4 * d * d)
    return loss


def pairwise_coral(feat_by_country):
    """Mean pairwise CORAL across the source countries for one feature set."""
    countries = list(feat_by_country.keys())
    val = 0.0
    pairs = 0
    for i in range(len(countries)):
        for j in range(i + 1, len(countries)):
            val = val + coral_loss(feat_by_country[countries[i]], feat_by_country[countries[j]])
            pairs += 1
    if pairs > 0:
        val = val / pairs
    return val


def stratified_random_boolean_tensor(n, batch_size, device, labels):
    assert len(labels) == n, "The length of labels must match n."
    assert batch_size <= n, "Batch size cannot be larger than the number of available elements."

    bool_tensor = torch.zeros(n, dtype=torch.bool)
    indices_0 = torch.where(labels == 0)[0]
    indices_1 = torch.where(labels == 1)[0]
    batch_size_0 = batch_size // 2
    batch_size_1 = batch_size - batch_size_0

    batch_size_0 = min(batch_size_0, len(indices_0))
    batch_size_1 = min(batch_size_1, len(indices_1))

    sampled_indices_0 = indices_0[torch.randperm(len(indices_0))[:batch_size_0]]
    sampled_indices_1 = indices_1[torch.randperm(len(indices_1))[:batch_size_1]]
    bool_tensor[sampled_indices_0] = True
    bool_tensor[sampled_indices_1] = True
    return bool_tensor.to(device)


def main(dataset_name, train_hyperparams, model_hyperparams, hyper_params, device_id):
    if model_hyperparams is None:
        model_hyperparams = DEFAULT_MODEL_HYPERPARAMETERS
    if train_hyperparams is None:
        train_hyperparams = DEFAULT_TRAIN_HYPERPARAMETERS
    if hyper_params is None:
        hyper_params = DEFAULT_HYPERPARAMETERS

    set_seed(hyper_params['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = device_id

    (device, base_dir, interim_data_dir, data_dir, datasets, edge_indices, network,
     node_features, struct_node_features, coverage_mask) = read_all_data_csdfa(
        device_id, dataset_name, hyper_params, train_hyperparams, model_hyperparams)

    model_hyperparams['num_textual_features'] = node_features.shape[1]
    model_hyperparams['num_structural_features'] = struct_node_features.shape[1]

    prior_lambda = train_hyperparams.get('coverage_prior_lambda', 1.0)
    target_log_prior = coverage_log_prior(coverage_mask)

    # Report the target country's channel coverage -- this is the failure-analysis
    # number the whole method keys off, so make it visible in the log.
    target_cov_frac = coverage_mask.mean(dim=0).cpu().numpy()
    print(f"\n[CS-DFA] Target '{dataset_name}' channel coverage: " +
          ", ".join(f"{s}={target_cov_frac[i]:.4f}" for i, s in enumerate(SUBNETS)))
    print(f"[CS-DFA] coverage_prior_lambda={prior_lambda}")

    # Preload source country datasets (data + per-node coverage masks + priors)
    other_countries = [c for c in ALL_COUNTRIES if c != dataset_name]
    countries_data = {}
    countries_numExamples = {}
    for country in other_countries:
        (_, _, _, _, c_datasets, c_edge_indices, _, c_node_features,
         c_struct_node_features, c_coverage_mask) = read_all_data_csdfa(
            device_id, country, hyper_params, train_hyperparams, model_hyperparams)
        countries_data[country] = {
            'datasets': c_datasets, 'edge_indices': c_edge_indices,
            'node_features': c_node_features, 'struct_node_features': c_struct_node_features,
            'coverage_mask': c_coverage_mask, 'log_prior': coverage_log_prior(c_coverage_mask)
        }
        countries_numExamples[country] = c_struct_node_features.shape[0]

    train_logger = TrainLogMetrics(hyper_params['num_splits'], ['supervised'])
    val_logger = TestLogMetrics(hyper_params['num_splits'], ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger = TestLogMetrics(hyper_params['num_splits'], ['accuracy', 'precision', 'f1_macro', 'f1_micro'])
    test_logger_coRT = TestLogMetrics(hyper_params['num_splits'],
                                      ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger_coURL = TestLogMetrics(hyper_params['num_splits'],
                                       ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger_hashSeq = TestLogMetrics(hyper_params['num_splits'],
                                         ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger_fastRT = TestLogMetrics(hyper_params['num_splits'],
                                        ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])
    test_logger_tweetSim = TestLogMetrics(hyper_params['num_splits'],
                                          ['accuracy', 'precision', 'f1_macro', 'f1_micro', 'roc_auc'])

    # Subnet masks for per-subnet test evaluation (numpy, size = #graph nodes).
    coRT_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    coRT_mask[list(datasets['coRT'].nodes())] = True
    coURL_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    coURL_mask[list(datasets['coURL'].nodes())] = True
    hashSeq_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    hashSeq_mask[list(datasets['hashSeq'].nodes())] = True
    fastRT_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    fastRT_mask[list(datasets['fastRT'].nodes())] = True
    tweetSim_mask = np.full(shape=(datasets['graph'].number_of_nodes(),), fill_value=False)
    tweetSim_mask[list(datasets['tweetSim'].nodes())] = True

    numpy_labels = datasets['labels'].long().detach().cpu().numpy()
    num_epochs = train_hyperparams['num_epochs']
    metric_to_optimize = train_hyperparams['metric_to_optimize']

    # Loss: BCE by default (DFA inheritance); focal available for loss-routing.
    loss_type = train_hyperparams.get('loss_type', 'bce')
    focal_gamma = train_hyperparams.get('focal_gamma', 2.0)
    focal_alpha = train_hyperparams.get('focal_alpha', 0.75)
    if loss_type == 'focal':
        loss_fn = FocalLoss(gamma=focal_gamma, alpha=focal_alpha)
        print(f"[Loss] Focal Loss: gamma={focal_gamma}, alpha={focal_alpha}")
    else:
        loss_fn = torch.nn.BCELoss()
        print("[Loss] BCELoss (DFA inheritance)")

    # CORAL config. coral_on=channel applies CORAL to the stable channels'
    # embeddings; coral_on=text reproduces the DFA/TSET text-projection variant
    # (known no-op); none disables it. coral_weight default 1.0 (DFA scale).
    coral_on = train_hyperparams.get('coral_on', 'channel')
    coral_weight = train_hyperparams.get('coral_weight', 1.0)
    coral_channel_names = train_hyperparams.get('coral_channels', ['coRT', 'coURL'])
    coral_channel_idx = [SUBNETS.index(c) for c in coral_channel_names]
    print(f"[CORAL] on={coral_on}, weight={coral_weight}, channels={coral_channel_names}")

    # Gating switch. gating=False -> plain per-node attention over all 5 channels
    # (no coverage mask, no prior): the AMC-style multichannel backbone control.
    gating = train_hyperparams.get('gating', 'on') == 'on'
    print(f"[Fusion] coverage gating = {gating}")

    for run_id in tqdm(range(hyper_params['num_splits']), 'Splits training'):
        BEST_VAL_METRIC = -np.inf
        best_model_path = interim_data_dir / f'model{run_id}.pth'
        model = create_model(model_hyperparams)
        model.to(device)

        optimizer = torch.optim.Adam(list(model.parameters()), lr=train_hyperparams['learning_rate'])
        early_stopping_cnt = 0

        for epoch in range(num_epochs):
            if early_stopping_cnt > train_hyperparams["early_stopping_limit"]:
                break
            model.train()
            optimizer.zero_grad()

            loss = 0
            task_loss_sum = 0
            text_features_dict = {}
            # channel_features_dict[channel_idx][country] = stable channel embeds
            channel_features_dict = {ci: {} for ci in coral_channel_idx}

            for country in countries_data:
                cd = countries_data[country]
                train_mask = stratified_random_boolean_tensor(
                    countries_numExamples[country],
                    batch_size=128, device=device,
                    labels=cd['datasets']['labels']
                )
                pred, text_feats, z_list = model(
                    cd['node_features'], cd['struct_node_features'], cd['edge_indices'],
                    cd['coverage_mask'], log_prior=cd['log_prior'], prior_lambda=prior_lambda,
                    gating=gating, return_text_features=True, return_channels=True
                )
                text_features_dict[country] = text_feats[train_mask]
                for ci in coral_channel_idx:
                    channel_features_dict[ci][country] = z_list[ci][train_mask]

                task_loss = loss_fn(pred.flatten()[train_mask],
                                    cd['datasets']['labels'][train_mask])
                loss += task_loss
                task_loss_sum += task_loss.item()

            # --- CORAL: channel-specific on stable channels, with text-proj as a
            #     side-by-side diagnostic (both logged, only one enters the loss).
            text_coral = pairwise_coral(text_features_dict)
            channel_coral_per = {ci: pairwise_coral(channel_features_dict[ci]) for ci in coral_channel_idx}
            channel_coral = sum(channel_coral_per.values()) / max(len(channel_coral_per), 1)

            if coral_on == 'channel':
                loss = loss + channel_coral * coral_weight
            elif coral_on == 'text':
                loss = loss + text_coral * coral_weight
            # coral_on == 'none' -> no alignment term

            loss.backward()
            optimizer.step()
            train_logger.train_update(run_id, 'supervised', loss.item())

            if epoch % train_hyperparams["check_loss_freq"] == 0:
                model.eval()
                with torch.no_grad():
                    pred = model(node_features, struct_node_features, edge_indices,
                                 coverage_mask, log_prior=target_log_prior,
                                 prior_lambda=prior_lambda, gating=gating).detach().cpu().numpy().flatten()
                    val_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['val'])
                    train_logger.val_update(run_id, val_metrics[train_hyperparams["metric_to_optimize"]])

                    if val_metrics[train_hyperparams["metric_to_optimize"]] > BEST_VAL_METRIC:
                        BEST_VAL_METRIC = val_metrics[train_hyperparams["metric_to_optimize"]]
                        torch.save(model.state_dict(), best_model_path)
                        early_stopping_cnt = 0
                    else:
                        early_stopping_cnt += 1

                    ch_str = ", ".join(f"{SUBNETS[ci]}={channel_coral_per[ci].item():.6f}"
                                       for ci in coral_channel_idx)
                    # text vs channel CORAL side by side -> verifies whether the
                    # channel-z alignment is non-trivial where text projection is a no-op.
                    print(f'Epoch {epoch}/{num_epochs} loss: {loss.item():.4f} '
                          f'(task: {task_loss_sum:.4f}) -- '
                          f'CORAL[text={text_coral.item():.6f} | channel_mean={channel_coral.item():.6f} | {ch_str}] '
                          f'-- val_{metric_to_optimize}: {val_metrics[metric_to_optimize]:.4f}')
            else:
                train_logger.val_update(run_id, 0.0)

        model.load_state_dict(torch.load(best_model_path, map_location=device))
        model.eval()
        with torch.no_grad():
            pred, final_weights = model(node_features, struct_node_features, edge_indices,
                                        coverage_mask, log_prior=target_log_prior,
                                        prior_lambda=prior_lambda, gating=gating, return_attention=True)
            pred = pred.detach().cpu().numpy().flatten()
            mean_weights = final_weights.mean(dim=0).cpu().numpy()
            print(f"\n[Split {run_id}] Mean channel attention (gated): " +
                  ", ".join(f"{s}={mean_weights[i]:.4f}" for i, s in enumerate(SUBNETS)))

        val_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['val'])
        for metric_name in val_metrics:
            val_logger.update(metric_name, run_id, val_metrics[metric_name])

        test_metrics = eval_pred(numpy_labels, pred > 0.5, datasets['splits'][run_id]['test'])
        for metric_name in test_metrics:
            test_logger.update(metric_name, run_id, test_metrics[metric_name])

        test_metrics_coRT = eval_pred(numpy_labels, pred > 0.5,
                                      np.logical_and(datasets['splits'][run_id]['test'], coRT_mask),
                                      prob_pred=pred)
        for metric_name in test_metrics_coRT:
            test_logger_coRT.update(metric_name, run_id, test_metrics_coRT[metric_name])

        test_metrics_coURL = eval_pred(numpy_labels, pred > 0.5,
                                       np.logical_and(datasets['splits'][run_id]['test'], coURL_mask),
                                       prob_pred=pred)
        for metric_name in test_metrics_coURL:
            test_logger_coURL.update(metric_name, run_id, test_metrics_coURL[metric_name])

        test_metrics_hashSeq = eval_pred(numpy_labels, pred > 0.5,
                                         np.logical_and(datasets['splits'][run_id]['test'], hashSeq_mask),
                                         prob_pred=pred)
        for metric_name in test_metrics_hashSeq:
            test_logger_hashSeq.update(metric_name, run_id, test_metrics_hashSeq[metric_name])

        test_metrics_fastRT = eval_pred(numpy_labels, pred > 0.5,
                                        np.logical_and(datasets['splits'][run_id]['test'], fastRT_mask),
                                        prob_pred=pred)
        for metric_name in test_metrics_fastRT:
            test_logger_fastRT.update(metric_name, run_id, test_metrics_fastRT[metric_name])

        test_metrics_tweetSim = eval_pred(numpy_labels, pred > 0.5,
                                          np.logical_and(datasets['splits'][run_id]['test'], tweetSim_mask),
                                          prob_pred=pred)
        for metric_name in test_metrics_tweetSim:
            test_logger_tweetSim.update(metric_name, run_id, test_metrics_tweetSim[metric_name])

    for split_num in tqdm(range(hyper_params['num_splits']), 'Splits post-training'):
        mlflow.log_artifact(interim_data_dir / f'model{split_num}.pth')

    save_metrics(val_logger, interim_data_dir, 'VAL')
    save_metrics(test_logger, interim_data_dir, 'TEST')
    save_metrics(test_logger_coRT, interim_data_dir, 'TEST_coRT')
    save_metrics(test_logger_coURL, interim_data_dir, 'TEST_coURL')
    save_metrics(test_logger_hashSeq, interim_data_dir, 'TEST_hashSeq')
    save_metrics(test_logger_fastRT, interim_data_dir, 'TEST_fastRT')
    save_metrics(test_logger_tweetSim, interim_data_dir, 'TEST_tweetSim')

    update_best_model_snapshot(data_dir, metric_to_optimize, test_logger,
                               hyper_params['num_splits'], interim_data_dir)

    print(f"\n[CS-DFA Summary for {dataset_name}]")
    print(f"  channel coverage = " +
          ", ".join(f"{s}={target_cov_frac[i]:.4f}" for i, s in enumerate(SUBNETS)))
    print(f"  loss_type={loss_type}, coral_on={coral_on}, coral_weight={coral_weight}, "
          f"coverage_prior_lambda={prior_lambda}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Run CS-DFA-GFM: Channel-Specific Decoupled Feature Alignment\n"
                    "  - 5 per-subnet ChannelGNNs (AMC/DMC backbone)\n"
                    "  - coverage-gated per-node attention fusion + coverage prior\n"
                    "  - BCE loss (DFA inheritance); --loss_type focal for loss-routing\n"
                    "  - channel-specific CORAL on stable channels (with text-proj diagnostic)",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-dataset_name', '--dataset', type=str, default='cuba')
    parser.add_argument('-seed', '--seed', type=int, default=12121995)
    parser.add_argument('-train_perc', '--train', type=float, default=.6)
    parser.add_argument('-val_perc', '--val', type=float, default=.2)
    parser.add_argument('-test_perc', '--test', type=float, default=.2)
    parser.add_argument('-num_splits', '--splits', type=int, default=5)
    parser.add_argument('-tweet_sim_threshold', '--tsim_th', type=float, default=.7)
    parser.add_argument('-device_id', '--device', type=str, default='0')
    parser.add_argument('-gnn_aggr_fn', '--aggr_fn', type=str, default='mean')
    parser.add_argument('-num_epochs', '--epochs', type=int, default=1000)
    parser.add_argument('-learning_rate', '--lr', type=float, default=1e-2)
    parser.add_argument('-early_stopping_limit', '--early', type=int, default=20)
    parser.add_argument('-check_loss_freq', '--check', type=int, default=1)
    parser.add_argument('-metric_to_optimize', '--val_metric', type=str, default='f1_macro')
    parser.add_argument('-gnn_type', '--gnn', type=str, default='sage')
    parser.add_argument('-gnn_embed_type', '--embed_type', type=str, default='positional_degree')
    parser.add_argument('-latent_dim', '--latent', type=int, default=128)
    parser.add_argument('-dropout', '--dropout', type=float, default=.2)
    parser.add_argument('-min_tweets', '--min_tweets', type=int, default=10)
    parser.add_argument('-most_popular', '--most_pop', type=int, default=5)
    parser.add_argument('-under_sampling', '--under', default=None)
    # CS-DFA-specific arguments
    parser.add_argument('-loss_type', '--loss_type', type=str, default='bce', choices=['bce', 'focal'],
                        help='Classification loss. bce (default, DFA) or focal (loss-routing).')
    parser.add_argument('-focal_gamma', '--focal_gamma', type=float, default=2.0)
    parser.add_argument('-focal_alpha', '--focal_alpha', type=float, default=0.75)
    parser.add_argument('-coverage_prior_lambda', '--cov_lambda', type=float, default=1.0,
                        help='Weight on the log-coverage prior in the fusion attention. '
                             '0 = pure gating; large = trust high-coverage channels.')
    parser.add_argument('-coral_on', '--coral_on', type=str, default='channel',
                        choices=['channel', 'text', 'none'],
                        help='Where to apply CORAL: stable channel embeds (default), '
                             'text projection (DFA no-op reproduction), or disabled.')
    parser.add_argument('-coral_weight', '--coral_weight', type=float, default=1.0,
                        help='Weight on the CORAL alignment term (DFA scale = 1.0).')
    parser.add_argument('-coral_channels', '--coral_channels', type=str, default='coRT,coURL',
                        help='Comma-separated stable channels to align (subset of '
                             'coRT,coURL,hashSeq,fastRT,tweetSim).')
    parser.add_argument('-gating', '--gating', type=str, default='on', choices=['on', 'off'],
                        help="Coverage-gated fusion. off = plain per-node attention over all "
                             "channels (no mask, no prior) = AMC-style backbone control.")
    args = parser.parse_args()

    hyper_parameters = {
        'train_perc': args.train, 'val_perc': args.val, 'test_perc': args.test,
        'aggr_type': args.aggr_fn, 'num_splits': args.splits, 'seed': args.seed,
        'tsim_th': args.tsim_th, 'min_tweets': args.min_tweets, 'most_pop': args.most_pop,
        'input_embed': args.embed_type, 'trace_type': 'all',
        'undersampling': float(args.under) if args.under is not None else None
    }
    train_hyperparameters = {
        'num_epochs': args.epochs, 'learning_rate': args.lr,
        'early_stopping_limit': args.early, 'check_loss_freq': args.check,
        'metric_to_optimize': args.val_metric,
        'input_embed': args.embed_type, 'trace_type': 'all',
        'loss_type': args.loss_type, 'focal_gamma': args.focal_gamma, 'focal_alpha': args.focal_alpha,
        'coverage_prior_lambda': args.cov_lambda,
        'coral_on': args.coral_on, 'coral_weight': args.coral_weight,
        'coral_channels': [c.strip() for c in args.coral_channels.split(',') if c.strip()],
        'gating': args.gating
    }
    model_hyperparameters = {'gnn_type': args.gnn, 'latent_dim': args.latent, 'dropout': args.dropout}
    main(args.dataset, train_hyperparameters, model_hyperparameters, hyper_parameters, args.device)
