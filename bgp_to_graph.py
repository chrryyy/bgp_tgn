import json
import csv
from datetime import datetime
from pathlib import Path

def iso_to_unix(iso_time: str):
    dt = datetime.fromisoformat(iso_time.replace("Z", "+00:00"))
    return int(dt.timestamp())

def process_bgp_json_to_tgn(json_path, output_path):
    with open(json_path, 'r') as f:
        updates = json.load(f)

    edge_events = []
    edge_id = 0

    for update in updates:
        if update["type"] != "announcement":
            continue  # TGN focuses on positive edge events

        timestamp = iso_to_unix(update["timestamp"])
        as_path = update.get("as_path", [])
        if len(as_path) < 2:
            continue  # no edge to form

        for i in range(len(as_path) - 1):
            src = as_path[i]
            dst = as_path[i + 1]
            event = {
                "source_node": src,
                "destination_node": dst,
                "timestamp": timestamp,
                "edge_idx": edge_id,
                "label": 1,  # CHANGE THIS DEPENDING ON DATA 0 for normal, 1 for anomaly (set externally)
                "node_features": 0.0,  # placeholder
                "edge_features": 1.0   # frequency or confidence
            }
            edge_events.append(event)
            edge_id += 1

    # Write CSV for TGN
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = [
            "source_node", "destination_node", "timestamp",
            "edge_idx", "label", "node_features", "edge_features"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(edge_events)

    print(f"Saved {len(edge_events)} TGN-formatted events to {output_path}")

# CHANGE THIS
process_bgp_json_to_tgn("./data/ttnet_anomalies.json", "./data/ttnet_anomalies.csv")
