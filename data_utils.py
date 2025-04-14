import pandas as pd
import numpy as np

def load_bgp_data(split="all", num_nodes=10000, node_feat_dim=100, edge_feat_dim=10):
    if split == "train":
        path = "./data/bgp/train.csv"
    elif split == "val":
        path = "./data/bgp/val.csv"
    elif split == "test":
        path = "./data/bgp/test.csv"
    else:
        path = "./data/bgp/tgn_events.csv"

    df = pd.read_csv(path)

    sources = df["source_node"].values
    destinations = df["destination_node"].values
    timestamps = df["timestamp"].values
    edge_idxs = df["edge_idx"].values
    labels = df["label"].values

    max_node_id = max(np.max(sources), np.max(destinations))
    num_nodes = max(num_nodes, max_node_id + 1)

    # Dummy node features with specified dimension
    node_feats = np.zeros((num_nodes, node_feat_dim), dtype=np.float32)

    # Dummy edge features with specified dimension
    edge_feats = np.ones((len(df), edge_feat_dim), dtype=np.float32)

    return sources, destinations, timestamps, edge_idxs, labels, node_feats, edge_feats

