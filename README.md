# SCALE_sim_v2_notes
Notes and hands-on for SCALE-Sim v2 (systolic array accelerator simulator).

## 📌 What is SCALE-Sim v2?
[SCALE-Sim v2](https://github.com/scalesim-project/scale-sim-v2) is a simulator for Systolic Array-based deep learning accelerators.  
It supports CNNs, feed-forward layers, and any workload that can be expressed as GEMM (General Matrix Multiplication) operations.  
It is useful for analyzing hardware performance and memory bandwidth requirements.

## ⚙ Installation and Basic Usage
SCALE-Sim v2 is written entirely in Python. You can install it via `pip` or run it directly from the source.

### Method 1: Install via pip
```bash
pip3 install scalesim
```

### Method 2: Install from source
git clone https://github.com/scalesim-project/scale-sim-v2.git
cd scale-sim-v2
python3 setup.py install

▶ Running a Simulation
SCALE-Sim requires two input files:

Config file – specifies architecture parameters and simulation settings.

Topology file – describes the workload (e.g., CNN layers or GEMM parameters).

Example:
```bash
python3 scale.py -c configs/eyeriss.cfg -t topologies/alexnet.csv -p ./outputs
```
The output directory will include:

COMPUTE_REPORT.csv – compute cycles, stall cycles, utilization percentages.

BANDWIDTH_REPORT.csv – average and peak SRAM/DRAM bandwidth usage.

DETAILED_ACCESS_REPORT.csv – access counts and cycles for each operand in SRAM and DRAM.

## 🧪 My Observations and Experience

**Experiment setup:**
- **Array size:** High (weight = 255 × 255)
- **SRAM sizes:** IFMAP = 8192 MB, FILTER = 8192 MB, OFMAP = 8192 MB
- **Memory offsets:** IFMAP = 0, FILTER = 10,000,000, OFMAP = 20,000,000
- **Dataflow:** WS (Weight Stationary)
- **Bandwidth:** 34
- **Memory banks:** 4
- **Interface bandwidth mode:** CALC

---

### 📊 MobileNet (224×224 input)

**Performance Summary**
| LayerID | Total Cycles | Stall Cycles | Overall Util % | Mapping Efficiency % | Compute Util % |
|---------|--------------|--------------|----------------|----------------------|----------------|
| 0 | 13309 | 0 | 1.2426 | 1.3184 | 1.2191 |
| 1 | 25731 | 0 | 0.2067 | 0.2197 | 0.2026 |
| 2 | 13309 | 0 | 2.9454 | 3.1250 | 2.8898 |
| 3 | 11705 | 0 | 0.2355 | 0.2930 | 0.2210 |
| 4 | 3901  | 0 | 10.0487 | 12.5000 | 9.4299 |

**Bandwidth Usage**
| LayerID | Avg IFMAP SRAM BW | Avg FILTER SRAM BW | Avg OFMAP SRAM BW | Avg IFMAP DRAM BW | Avg FILTER DRAM BW | Avg OFMAP DRAM BW |
|---------|-------------------|--------------------|-------------------|-------------------|--------------------|-------------------|
| 0 | 25.4480 | 0.0649 | 30.1606 | 8.9717 | 0.1030 | 256.0000 |
| 1 | 135.4320 | 0.0112 | 0.9405 | 9.5703 | 0.0343 | 254.7368 |
| 2 | 30.1606 | 0.1539 | 60.3213 | 9.5703 | 0.2441 | 256.0000 |
| 3 | 154.3217 | 0.0492 | 0.8038 | 9.5703 | 0.0687 | 254.2703 |
| 4 | 51.4494 | 2.1000 | 102.8987 | 7.9752 | 0.9765 | 256.0000 |

**Layer Configurations**
| LayerID | Layer name | IFMAP Height | IFMAP Width | Filter Height | Filter Width | Channels | Num Filter | Strides |
|---------|------------|--------------|-------------|---------------|--------------|----------|------------|---------|
| 0 | Conv1 | 224 | 224 | 3 | 3 | 3 | 32 | 2 |
| 1 | Conv2 | 112 | 112 | 3 | 3 | 32 | 1 | 1 |
| 2 | Conv3 | 112 | 112 | 1 | 1 | 32 | 64 | 1 |
| 3 | Conv4 | 112 | 112 | 3 | 3 | 64 | 1 | 2 |
| 4 | Conv5 | 56  | 56  | 1 | 1 | 64 | 128 | 1 |

---

### 📊 GPT-2 Model Components

**Performance Summary**
| LayerID | Total Cycles | Stall Cycles | Overall Util % | Mapping Efficiency % | Compute Util % |
|---------|--------------|--------------|----------------|----------------------|----------------|
| 0 | 238069 | 0 | 50.4056 | 88.1109 | 44.1201 |
| 1 | 7159   | 0 | 14.3037 | 25.0000 | 12.5183 |
| 2 | 7159   | 0 | 14.3037 | 25.0000 | 12.5183 |
| 3 | 87709  | 0 | 45.6054 | 79.7194 | 39.9182 |
| 4 | 150359 | 0 | 51.0778 | 89.2857 | 44.7083 |

**Bandwidth Usage**
| LayerID | Avg IFMAP SRAM BW | Avg FILTER SRAM BW | Avg OFMAP SRAM BW | Avg IFMAP DRAM BW | Avg FILTER DRAM BW | Avg OFMAP DRAM BW |
|---------|-------------------|--------------------|-------------------|-------------------|--------------------|-------------------|
| 0 | 130.7587 | 32.2596 | 144.5228 | 9.7656 | 14.0492 | 162.1067 |
| 1 | 36.6174  | 9.1544  | 146.4696 | 7.8121 | 7.8121  | 256.0000 |
| 2 | 146.4696 | 9.1544  | 36.6174  | 9.6154 | 7.8121  | 256.0000 |
| 3 | 130.7597 | 29.1874 | 130.7597 | 9.7656 | 9.8444  | 162.4719 |
| 4 | 130.7591 | 32.6898 | 146.4501 | 9.7656 | 8.9915  | 175.7340 |

**Layer Configurations**
| Layer name | IFMAP Height | IFMAP Width | Filter Height | Filter Width | Channels | Num Filter | Strides |
|------------|--------------|-------------|---------------|--------------|----------|------------|---------|
| Linear1    | 1024 | 1600 | 1 | 1600 | 1 | 4800  | 1 |
| QKT        | 1024 | 64   | 1 | 64   | 1 | 1024  | 1 |
| QKTV       | 1024 | 1024 | 1 | 1024 | 1 | 64    | 1 |
| Linear2    | 1024 | 1600 | 1 | 1600 | 1 | 1600  | 1 |
| PW-FF-L1   | 1024 | 1600 | 1 | 1600 | 1 | 3072  | 1 |

---

### 💡 Insights
- **Dataflow choice is critical:** WS mode performs well for dense GEMM operations (GPT-2) but poorly for depthwise convolutions (MobileNet).
- **OFMAP DRAM bottleneck:** In MobileNet, OFMAP writes saturate DRAM bandwidth (256) in most layers, indicating a write-back bottleneck.
- **Utilization gap:** The same hardware configuration shows drastically different utilization between CNN and Transformer workloads, emphasizing the need for workload-specific hardware tuning.

