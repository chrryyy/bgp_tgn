import math
import logging
import time
import sys
import random
import argparse
import pickle
from pathlib import Path

import torch
import numpy as np

from model.tgn import TGN
from utils.utils import EarlyStopMonitor, get_neighbor_finder, MLP
from utils.data_processing import compute_time_statistics
from data_utils import load_bgp_data  # <- ✅ use your own loader here

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

from types import SimpleNamespace

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


parser = argparse.ArgumentParser('TGN training on BGP data')
parser.add_argument('--bs', type=int, default=100, help='Batch size')
parser.add_argument('--prefix', type=str, default='bgp', help='Run name prefix')
parser.add_argument('--n_degree', type=int, default=10)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--n_epoch', type=int, default=10)
parser.add_argument('--n_layer', type=int, default=1)
parser.add_argument('--lr', type=float, default=3e-4)
parser.add_argument('--patience', type=int, default=5)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--node_dim', type=int, default=100)
parser.add_argument('--time_dim', type=int, default=100)
parser.add_argument('--use_memory', action='store_true')
parser.add_argument('--embedding_module', type=str, default="graph_attention")
parser.add_argument('--message_function', type=str, default="identity")
parser.add_argument('--aggregator', type=str, default="last")
parser.add_argument('--memory_update_at_end', action='store_true')
parser.add_argument('--message_dim', type=int, default=100)
parser.add_argument('--memory_dim', type=int, default=172)
args = parser.parse_args()

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-bgp-tgn.pth'

logger = logging.getLogger()
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.DEBUG)

logger.info("Loading BGP data...")
train_data = tuple_to_namespace(load_bgp_data("train"))
val_data = tuple_to_namespace(load_bgp_data("val"))
test_data = tuple_to_namespace(load_bgp_data("test"))

sources = train_data.sources
destinations = train_data.destinations
timestamps = train_data.timestamps
edge_idxs = train_data.edge_idxs
labels = train_data.labels
node_features = train_data.node_features
edge_features = train_data.edge_features

max_idx = max(np.max(sources), np.max(destinations))

train_ngh_finder = get_neighbor_finder(train_data, uniform=False, max_node_idx=max_idx)

device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
    compute_time_statistics(sources, destinations, timestamps)

logger.info("Initializing TGN model...")
tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
          edge_features=edge_features, device=device,
          n_layers=args.n_layer, n_heads=args.n_head, dropout=args.drop_out,
          use_memory=args.use_memory, message_dimension=args.message_dim,
          memory_dimension=args.memory_dim, memory_update_at_start=not args.memory_update_at_end,
          embedding_module_type=args.embedding_module,
          message_function=args.message_function, aggregator_type=args.aggregator,
          n_neighbors=args.n_degree, mean_time_shift_src=mean_time_shift_src,
          std_time_shift_src=std_time_shift_src,
          mean_time_shift_dst=mean_time_shift_dst,
          std_time_shift_dst=std_time_shift_dst)

tgn = tgn.to(device)

num_instance = len(sources)
num_batch = math.ceil(num_instance / args.bs)

logger.info(f"Training with {num_instance} samples across {num_batch} batches")

decoder = MLP(node_features.shape[1], drop=args.drop_out).to(device)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr)
decoder_loss_criterion = torch.nn.BCELoss()
early_stopper = EarlyStopMonitor(max_round=args.patience)

for epoch in range(args.n_epoch):
    if args.use_memory:
        tgn.memory.__init_memory__()

    tgn.eval()
    decoder.train()
    epoch_loss = 0

    for k in range(num_batch):
        s_idx = k * args.bs
        e_idx = min(num_instance, s_idx + args.bs)

        src_batch = sources[s_idx:e_idx]
        dst_batch = destinations[s_idx:e_idx]
        ts_batch = timestamps[s_idx:e_idx]
        edge_idx_batch = edge_idxs[s_idx:e_idx]
        label_batch = labels[s_idx:e_idx]

        decoder_optimizer.zero_grad()

        with torch.no_grad():
            src_emb, dst_emb, _ = tgn.compute_temporal_embeddings(
                src_batch, dst_batch, dst_batch, ts_batch, edge_idx_batch, args.n_degree
            )
        decoder = MLP(input_dim=src_emb.shape[1], drop=0.1)

        pred = decoder(src_emb).sigmoid()
        loss = decoder_loss_criterion(pred, torch.FloatTensor(label_batch).to(device))
        loss.backward()
        decoder_optimizer.step()

        epoch_loss += loss.item()

    logger.info(f"Epoch {epoch}: train loss = {epoch_loss / num_batch:.4f}")

    # Validation AUC
    val_auc = 0  # <- plug in your custom eval here if desired
    if early_stopper.early_stop_check(val_auc):
        logger.info("Early stopping triggered")
        break

torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
logger.info("Training completed. Model saved.")
