# ONOS-GNN-PPO: AI-Driven Secure and Energy-Aware Routing for Healthcare IoT Networks

This repository accompanies the paper:
> **"ONOS-GNN-PPO: AI-Driven Secure and Energy-Aware Routing Framework for Healthcare IoT Networks"**  
> Submitted to *IEEE Transactions on Network and Service Management (TNSM), 2025.*


🧩 Overview
The framework integrates:
- **Graph Neural Networks (GNN):** topology-aware routing
- **Proximal Policy Optimization (PPO):** adaptive and energy-efficient control
- **Federated Learning (FL):** privacy-preserving anomaly detection
- **ONOS Controller:** centralized SDN intelligence

Designed for secure, scalable, and energy-optimized Healthcare IoT (H-IoT) systems.


⚙️ Quick Start
```bash
pip install -r requirements.txt
bash run_experiment.sh
````

Software Environment

* Mininet 2.3.0
* ONOS v2.6
* Python 3.9
* Ubuntu 22.04
* TensorFlow / PyTorch for model training


📊 Results

Ready figures are stored in `/results/`:

* Throughput, Latency, Energy, ROC, Ablation, and λ-Sensitivity plots.
  All results averaged over 10 runs for statistical significance (95% CI).


🔒 Dataset

This framework uses the **H-CIoT dataset** (Normal, DoS, ARP, PortScan, Smurf).
Due to privacy constraints, only synthetic samples are provided for reproduction.
See `/dataset/README.txt` for structure and ethics statement.


🧠 Citation

```
@article{ONOSGNNPPO2025,
  author    = {Anil Ram and Swarnendu Kumar Chakraborty},
  title     = {ONOS-GNN-PPO: AI-Driven Secure and Energy-Aware Routing Framework for Healthcare IoT Networks},
  journal   = {IEEE Transactions on Network and Service Management},
  year      = {2025}
}
```





📜 License

This project is released under the MIT License.
