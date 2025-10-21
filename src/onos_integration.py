"""
onos_integration.py
Placeholder helper to demonstrate how ONOS integration would be scripted.
- Contains a 'simulation' mode (no actual ONOS calls) for reproducible demos.
- Also provides a REST client template for pushing flow rules to ONOS.

Usage:
    python onos_integration.py --mode simulate
    python onos_integration.py --mode push_flow --flow_json path/to/flow.json

Note:
  For real deployments, set ONOS_HOST, ONOS_USER, ONOS_PASS and use the REST endpoints.
  This script intentionally does not perform destructive operations.
"""

import argparse
import json
import time
import requests  # optional if doing real calls; safe to be present
from utils import set_seed

set_seed(0)


ONOS_HOST = "http://127.0.0.1:8181"  # change for real deployment
ONOS_USER = "onos"
ONOS_PASS = "rocks"


def simulate_workflow():
    print("Simulating ONOS workflow (no real controller calls).")
    for step in ["collect topology", "compute GNN embeddings", "compute PPO actions", "install flows"]:
        print(f" - {step} ...")
        time.sleep(0.3)
    print("Simulation complete. Example flow rules saved to results/flow_sample.json")
    # Example sample flow (dummy)
    sample = {
        "deviceId": "of:0000000000000001",
        "treatment": {"instructions": [{"type": "OUTPUT", "port": "2"}]},
        "selector": {"criteria": [{"type": "ETH_TYPE", "ethType": "0x0800"}]}
    }
    with open("results/flow_sample.json", "w") as f:
        json.dump(sample, f, indent=2)


def push_flow_to_onos(flow_json_path):
    # REST API template - make sure ONOS is reachable and credentials set
    url = f"{ONOS_HOST}/onos/v1/flows"
    with open(flow_json_path, "r") as f:
        payload = json.load(f)
    print(f"Pushing flow to ONOS: {url}")
    # For safety, do not actually call in demo mode; commented out:
    # resp = requests.post(url, auth=(ONOS_USER, ONOS_PASS), json=payload)
    # print("Response:", resp.status_code, resp.text)
    print("Functionality stubbed - enable real calls by uncommenting requests.post.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["simulate", "push_flow"], default="simulate")
    parser.add_argument("--flow_json", type=str, default="results/flow_sample.json")
    args = parser.parse_args()
    if args.mode == "simulate":
        simulate_workflow()
    elif args.mode == "push_flow":
        push_flow_to_onos(args.flow_json)
