===========================================================
H-CIoT Dataset Documentation
===========================================================

Title: Healthcare IoT (H-CIoT) Network Traffic Dataset
Version: 1.0 (Modified for ONOS-GNN-PPO Framework)
Authors: Research Group - ONOS-GNN-PPO Project, 2025
Contact: yourname@domain.com

-----------------------------------------------------------
 1. Overview
-----------------------------------------------------------
The H-CIoT dataset is used to evaluate anomaly detection, routing, and
energy efficiency in Software-Defined Networking (SDN)-based Healthcare IoT (H-IoT) environments.

It contains labeled traffic traces from simulated healthcare devices 
(sensors, wearables, gateways) operating over an SDN architecture with
five traffic categories:

1. Normal Traffic
2. Denial-of-Service (DoS)
3. ARP Spoofing
4. PortScan
5. Smurf Attacks

The dataset exhibits natural class imbalance to emulate real-world
healthcare scenarios where rare attacks are infrequent but highly critical.

-----------------------------------------------------------
 2. File Structure
-----------------------------------------------------------
This folder contains only metadata and synthetic feature samples for 
reproducibility. The complete dataset used in experiments is described 
below:

- **HCIoT_Features.csv** (not included here due to privacy constraints)
  • 50,000 normal samples
  • 12,500 DoS samples
  • 6,000 ARP spoofing samples
  • 8,000 PortScan samples
  • 3,500 Smurf samples

For ethical and privacy compliance, only non-identifiable synthetic
feature samples and statistical summaries can be shared publicly.

-----------------------------------------------------------
 3. Data Format
-----------------------------------------------------------
Each record in the dataset corresponds to one network flow with
the following attributes:

| Feature | Description |
|----------|--------------|
| src_ip | Source device IP address |
| dst_ip | Destination device IP address |
| pkt_rate | Packet rate (packets/sec) |
| avg_delay | Average delay (ms) |
| jitter | Variation in packet delay |
| flow_bytes | Total bytes transferred |
| cpu_usage | Local node CPU utilization (%) |
| mem_usage | Local node memory utilization (%) |
| label | Traffic type (Normal/DoS/ARP/PortScan/Smurf) |

-----------------------------------------------------------
 4. Ethical and Privacy Considerations
-----------------------------------------------------------
• The dataset does NOT contain any patient identifiers or sensitive
  healthcare information.
• Traffic traces were synthetically generated using healthcare communication 
  models over Mininet and ONOS emulation.
• All experiments comply with research ethics and privacy preservation
  guidelines outlined by IEEE and institutional review standards.

-----------------------------------------------------------
 5. Access and Reproducibility
-----------------------------------------------------------
For full dataset access:
- Refer to the corresponding author upon reasonable request.
- Alternatively, synthetic subset and feature statistics are provided
  for replication purposes within this repository.

Researchers can reproduce the results using:
1. The synthetic dataset sample.
2. Configuration files in `config/parameters.yaml`.
3. The model scripts in `src/`.

-----------------------------------------------------------
 6. Citation
-----------------------------------------------------------
If you use this dataset or the derived framework, please cite:

@article{ONOSGNNPPO2025,
  author    = {Your Name and Co-authors},
  title     = {ONOS-GNN-PPO: AI-Driven Secure and Energy-Aware Routing Framework for Healthcare IoT Networks},
  journal   = {IEEE Transactions on Network and Service Management},
  year      = {2025}
}

-----------------------------------------------------------
 7. Notes
-----------------------------------------------------------
• The dataset was used in Figures 1–6 of the paper.
• Imbalance mitigation used weighted cross-entropy loss during training.
• Synthetic traffic was generated via Mininet and ONOS using the
  topologies defined in `config/topology_geant.json` and `config/topology_nsfn.json`.

===========================================================
End of Dataset README
===========================================================
