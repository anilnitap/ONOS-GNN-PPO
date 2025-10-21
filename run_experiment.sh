#!/bin/bash
echo "==========================================="
echo " ONOS-GNN-PPO: Automated Experiment Runner "
echo "==========================================="

# Step 1: Environment Check
echo "[1/5] Checking Python environment..."
python3 --version
pip3 install -r requirements.txt

# Step 2: Load configuration
echo "[2/5] Loading experiment configuration..."
CONFIG_FILE="config/parameters.yaml"
echo "Using configuration from $CONFIG_FILE"

# Step 3: Initialize ONOS + Mininet
echo "[3/5] Launching ONOS and Mininet environment..."
sudo mn -c
sudo mn --controller=remote,ip=127.0.0.1,port=6653 --topo tree,depth=2,fanout=3 &

# Step 4: Run main scripts
echo "[4/5] Running GNN-PPO and Federated Learning modules..."
python3 src/gnn_model.py
python3 src/ppo_agent.py
python3 src/federated_ids.py
python3 src/onos_integration.py

# Step 5: Save results
echo "[5/5] Saving IEEE-style figures to /results directory..."
mkdir -p results
echo "All experiments complete. Results available in /results/"
