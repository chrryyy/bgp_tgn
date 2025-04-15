import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
from types import SimpleNamespace
import matplotlib.pyplot as plt
from IPython.display import clear_output

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import compute_time_statistics
from data_utils import load_bgp_data


def tuple_to_namespace(data_tuple):
    return SimpleNamespace(
        sources=data_tuple[0],
        destinations=data_tuple[1],
        timestamps=data_tuple[2],
        edge_idxs=data_tuple[3],
        labels=data_tuple[4],
        node_features=data_tuple[5],
        edge_features=data_tuple[6],
    )

# Argument and global variables
parser = argparse.ArgumentParser('TGN training on BGP data')
parser.add_argument('--bs', type=int, default=200)
parser.add_argument('--prefix', type=str, default='bgp')
parser.add_argument('--n_degree', type=int, default=10)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=50)
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--node_dim', type=int, default=100)
parser.add_argument('--time_dim', type=int, default=100)
parser.add_argument('--use_memory', action='store_true')
parser.add_argument('--embedding_module', type=str, default="graph_attention")
parser.add_argument('--message_function', type=str, default="identity")
parser.add_argument('--memory_updater', type=str, default="gru")
parser.add_argument('--aggregator', type=str, default="last")
parser.add_argument('--memory_update_at_end', action='store_true')
parser.add_argument('--message_dim', type=int, default=100)
parser.add_argument('--memory_dim', type=int, default=100)
parser.add_argument('--uniform', action='store_true', help='Use uniform sampling of neighbors')

args = parser.parse_args()

BATCH_SIZE = args.bs
DEVICE = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-tgn.pth'

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info(args)

# Load BGP data
logger.info("Loading BGP data")
train_data = tuple_to_namespace(load_bgp_data("train"))
val_data = tuple_to_namespace(load_bgp_data("val"))
test_data = tuple_to_namespace(load_bgp_data("test"))

new_node_val_data = val_data  # use same if no special new node split
new_node_test_data = test_data

node_features = train_data.node_features
edge_features = train_data.edge_features

full_data = SimpleNamespace(
    sources=np.concatenate([train_data.sources, val_data.sources, test_data.sources]),
    destinations=np.concatenate([train_data.destinations, val_data.destinations, test_data.destinations]),
    timestamps=np.concatenate([train_data.timestamps, val_data.timestamps, test_data.timestamps]),
    edge_idxs=np.concatenate([train_data.edge_idxs, val_data.edge_idxs, test_data.edge_idxs]),
)

# Neighbor finders
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Negative samplers
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
nn_val_rand_sampler = RandEdgeSampler(new_node_val_data.sources, new_node_val_data.destinations, seed=1)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)
nn_test_rand_sampler = RandEdgeSampler(new_node_test_data.sources, new_node_test_data.destinations, seed=3)

# Time stats
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

# Initialize model
logger.info("Initializing TGN model")
tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
          edge_features=edge_features, device=DEVICE,
          n_layers=args.n_layer, n_heads=args.n_head, dropout=args.drop_out,
          use_memory=args.use_memory, message_dimension=args.message_dim,
          memory_dimension=args.memory_dim,
          memory_update_at_start=not args.memory_update_at_end,
          embedding_module_type=args.embedding_module,
          message_function=args.message_function,
          aggregator_type=args.aggregator,
          memory_updater_type=args.memory_updater,
          n_neighbors=args.n_degree,
          mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
          mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst)

optimizer = torch.optim.Adam(tgn.parameters(), lr=args.lr)
criterion = torch.nn.BCELoss()
tgn = tgn.to(DEVICE)

# Training
logger.info("Training model now.")
early_stopper = EarlyStopMonitor(max_round=args.patience)
num_instance = len(train_data.sources)
num_batch = math.ceil(num_instance / BATCH_SIZE)

train_losses = []
val_aps = []

for epoch in range(args.n_epoch):
    if args.use_memory:
        tgn.memory.__init_memory__()

    tgn.set_neighbor_finder(train_ngh_finder)
    tgn.train()
    epoch_loss = []

    for k in range(num_batch):
        start_idx = k * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)

        src = train_data.sources[start_idx:end_idx]
        dst = train_data.destinations[start_idx:end_idx]
        ts = train_data.timestamps[start_idx:end_idx]
        eidx = train_data.edge_idxs[start_idx:end_idx]
        size = len(src)

        _, neg_dst = train_rand_sampler.sample(size)

        pos_label = torch.ones(size, device=DEVICE)
        neg_label = torch.zeros(size, device=DEVICE)

        pos_prob, neg_prob = tgn.compute_edge_probabilities(src, dst, neg_dst, ts, eidx, args.n_degree)
        loss = criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.item())

        if args.use_memory:
            tgn.memory.detach_memory()

    avg_loss = np.mean(epoch_loss)
    train_losses.append(avg_loss)
    logger.info(f"Epoch {epoch} | Loss: {avg_loss:.4f}")

    # Validation
    tgn.set_neighbor_finder(full_ngh_finder)
    if args.use_memory:
        train_mem = tgn.memory.backup_memory()

    val_ap, val_auc = eval_edge_prediction(tgn, val_rand_sampler, val_data, args.n_degree)
    val_aps.append(val_ap)
    logger.info(f"Validation AP: {val_ap:.4f}, AUC: {val_auc:.4f}")

    # Live plot
    clear_output(wait=True)
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_aps, label='Validation AP')
    plt.xlabel('Epoch')
    plt.title('Training Progress')
    plt.legend()
    plt.grid(True)
    plt.show()

    if early_stopper.early_stop_check(val_ap):
        logger.info(f"Early stopping at epoch {epoch}")
        break

# Save model
torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
logger.info("Training completed and model saved.")
