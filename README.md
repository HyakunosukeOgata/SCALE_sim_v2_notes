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
## 🧩 Design Suggestions from This Report (PE Array / SRAM / Workload Fit)

This section provides actionable recommendations for **PE array size & aspect ratio**, **SRAM sizing & partitioning**, and **workload suitability**, based on the observed results (low utilization in MobileNet, ~50% utilization in GPT-2, sustained OFMAP DRAM saturation at 256 in MobileNet).

---

### 1) PE Array: Size & Aspect Ratio
**Observation link**: MobileNet layers are dominated by depthwise separable and 1×1 convolutions → extremely low utilization; GPT-2 large GEMM layers achieve ~50% utilization.

- **For CNN/Depthwise (MobileNet)**  
  - Use a **smaller or partitionable array** to avoid massive PE idling when the number of channels per layer is small.  
    Example: reduce from 255×255 to **128×128** or **64×128**, or enable **multi-partition mode** to activate only part of the array dynamically.
  - Support **non-square (rectangular)** shapes: for 1×1 or low-channel layers, shapes like **64×256** or **128×256** can improve mapping efficiency by stretching spatial tiling along the longer dimension.
  - **Per-layer dataflow switching**: for depthwise, use OS/RS (Output-Stationary / Row-Stationary) to reduce OFMAP write-backs.

- **For Transformer/GEMM (GPT-2)**  
  - Match **matrix aspect ratio** with **configurable array shapes**:  
    - tall-skinny (M≫N) → **256×128**  
    - wide-short (N≫M) → **128×256**  
    - near-square → **256×256** or **192×192**
  - If external bandwidth is not a bottleneck, **increase total PE count** beyond 255×255 — but verify DRAM/NoC throughput.

> TL;DR: CNNs benefit from “partitionable + rectangular” arrays for small-channel layers; GEMM prefers “large + aspect-ratio matched” arrays.

---

### 2) SRAM: Capacity & Partitioning
**Observation link**: In MobileNet, **Avg OFMAP DRAM BW = 256** is saturated across many layers → output write-backs are the main bottleneck.

- **Prioritize expanding OFMAP SRAM (with double-buffering)**:  
  Goal: keep partial sums on-chip longer, reducing OFMAP write-back frequency.  
  Recommendation: given IFMAP/FILTER are both 8192 MB, **rebalance more capacity to OFMAP** if total SRAM budget cannot grow.
- **Increase SRAM bank count and multi-porting**:  
  Reduce bank conflicts and prevent on-chip write stalls.
- **Enable compression and write-combining**:  
  Apply zero compression or bit-width compression to OFMAP before DRAM write, and coalesce small writes at the DMA interface.

> Rule of thumb: if **Avg OFMAP DRAM BW** is persistently maxed out, first expand OFMAP SRAM/double-buffer, then consider increasing interface bandwidth.

---

### 3) Workload Suitability
- **Large GEMM / Transformer (GPT-2)**:  
  Well-suited for **large arrays** with WS/IS dataflows; utilization improves when array shape matches the matrix aspect ratio.  
  Once mapping efficiency is already high (~80–90%), check if **interface/DRAM/NoC** become the next limiting factor.
- **Depthwise / 1×1-heavy CNN (MobileNet)**:  
  Best served by **partitionable or smaller arrays**, OS/RS dataflows, and **larger OFMAP SRAM**. Otherwise, expect simultaneous **OFMAP DRAM saturation** and **PE idling**.

---

## 🏁 Overall Summary of This Experiment

This round of experiments with **SCALE-Sim v2** explored its ability to model different workloads — specifically **MobileNet (CNN)** and **GPT-2 (Transformer/GEMM)** — under the same hardware configuration:

- **Array size**: 255 × 255 (Weight-Stationary mode)
- **SRAM sizes**: IFMAP/FILTER/OFMAP each 8192 MB
- **Interface Bandwidth**: CALC mode, DRAM BW limit 256
- **Memory banks**: 4

### Key Findings
1. **Workload-dependent utilization gap**  
   - MobileNet depthwise & 1×1 convolution layers achieved extremely low overall utilization (<3% in many layers), despite having no stall cycles.  
   - GPT-2 GEMM layers achieved much higher utilization (~50%), benefiting from dense matrix operations.
   
2. **OFMAP DRAM saturation in MobileNet**  
   - Multiple layers hit the DRAM write bandwidth limit (256), indicating that output write-backs, not computation, were the primary bottleneck.

3. **Mapping efficiency vs. compute utilization**  
   - GPT-2 achieved high mapping efficiency (~80–90%) but compute utilization still lagged behind the theoretical maximum, suggesting room for hardware-array shape tuning.

4. **Same hardware, different needs**  
   - A single fixed hardware configuration cannot simultaneously maximize utilization for both depthwise CNN layers and large GEMM workloads without either array reconfiguration or dynamic dataflow switching.

### Reflections on SCALE-Sim v2 as a Tool
- **Strengths**:  
  - Flexible modeling of systolic array architectures with configurable size, dataflow, and memory hierarchy.  
  - Clear and detailed reports on cycles, utilization, and bandwidth, enabling precise bottleneck analysis.  
  - Supports both CNN and GEMM workloads, making it applicable for mixed AI models.

- **Limitations**:  
  - Does not directly model some micro-architectural optimizations (e.g., partial sum compression, bank conflict handling).  
  - Results rely on static configurations — dynamic reconfiguration needs to be tested by multiple runs with adjusted parameters.

- **Best use cases**:  
  - Early-stage design space exploration for AI accelerators.  
  - Comparative analysis between workloads (e.g., CNN vs. Transformer).  
  - Validating bandwidth and on-chip memory sizing before committing to RTL or hardware prototypes.

**Conclusion**:  
SCALE-Sim v2 proved to be a valuable tool for identifying **utilization gaps**, **memory bottlenecks**, and **dataflow mismatches** between workloads. While it does not replace cycle-accurate RTL simulation for final hardware validation, it significantly accelerates **design iteration and architectural decision-making** at the pre-implementation stage.
