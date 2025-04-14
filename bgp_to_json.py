import json
from datetime import datetime

def parse_bgp_line(line):
    parts = line.strip().split("|")
    if len(parts) < 7:
        return None  # Skip malformed lines

    timestamp_raw = parts[1].strip()
    try:
        timestamp = datetime.strptime(timestamp_raw, "%m/%d/%y %H:%M:%S").isoformat() + "Z"
    except ValueError:
        return None  # Skip lines with bad timestamps

    msg_type = parts[2].strip()
    collector_ip = parts[3].strip()
    peer_asn = int(parts[4].strip())
    prefix = parts[5].strip()
    as_path_tokens = parts[6].strip().split()
    as_path = [int(asn) for asn in as_path_tokens if not asn.startswith("{") and not asn.endswith("}")]
    origin = parts[7].strip() if len(parts) > 7 else None

    return {
        "timestamp": timestamp,
        "type": "announcement" if msg_type == "A" else "withdrawal",
        "collector_ip": collector_ip,
        "peer_asn": peer_asn,
        "prefix": prefix,
        "as_path": as_path,
        "origin": origin
    }

def process_bgp_file(input_path, output_path):
    parsed_events = []

    with open(input_path, 'r') as f:
        for line in f:
            if not line.strip().startswith("BGP4MP"):
                continue  # skip non-BGP4MP lines
            parsed = parse_bgp_line(line)
            if parsed:
                parsed_events.append(parsed)

    with open(output_path, 'w') as out:
        json.dump(parsed_events, out, indent=2)

    print(f"Processed {len(parsed_events)} valid BGP updates to {output_path}")

# CHANGE THIS LINE
process_bgp_file("./data/aws_anomalies.txt", "./data/aws_anomalies.json")
