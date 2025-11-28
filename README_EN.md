# Spurious Forgetting Identification and Mitigation Mechanism Experimental Code

## 📋 Table of Contents

- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Environment Setup](#environment-setup)
- [Running Experiments](#running-experiments)
- [Experiment Groups](#experiment-groups)
- [Experimental Results](#experimental-results)
- [Common Issues](#common-issues)
- [Project Structure](#project-structure)

---

## 🚀 Quick Start

### 3-Step Quick Run

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Install Ollama**: Visit https://ollama.ai to download and install, then run `ollama serve`
3. **Pull models**: `ollama pull qwen2.5:3b` (or use all 4 models)
4. **Run experiments**: `python run_experiments.py`

### Check Environment

```bash
# Check Ollama connection and models
python run_experiments.py --check-ollama
```

---

## Project Overview

This experimental codebase implements core experiments for spurious forgetting identification and mitigation mechanisms, including:

1. **Spurious Forgetting Identification Validation**: Validate the effectiveness of proposed spurious forgetting identification metrics and analysis tools
2. **Deep Alignment Training Validation**: Validate the effectiveness of Deep Alignment Training
3. **Ablation Study**: Analyze the contribution of each component and validate the rationality of method design

### Core Features

- ✅ Uses 4 Qwen models deployed via Ollama (1.7B, 3B, 4B, 32B)
- ✅ Implements Deep Alignment Training
- ✅ Implements Adaptive Mitigation Strategy
- ✅ Unified experiment entry point (`run_experiments.py`)

---

## Environment Setup

### Prerequisites

- Python 3.8+
- Ollama (for running Qwen models)
- CUDA (optional, for GPU acceleration)

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install and Configure Ollama

#### Windows:
1. Visit https://ollama.ai/download to download and install
2. Open command line and run:
```bash
ollama serve
```

#### Linux/Mac:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
ollama serve
```

### 3. Pull Models

```bash
ollama pull qwen3:1.7b
ollama pull qwen2.5:3b
ollama pull qwen3:4b
ollama pull qwen2.5:32b
```

**Note**: The 32B model is large (approximately 18GB). If resources are limited, you can start by pulling smaller models.

### 4. Check Ollama Setup

```bash
python run_experiments.py --check-ollama
```

---

## Running Experiments

### Basic Run

Run all experiments (using all 4 models and all experiment groups):

```bash
python run_experiments.py
```

### Specify Parameters

```bash
# Use specific models
python run_experiments.py --models qwen2.5:3b qwen3:4b

# Use specific datasets
python run_experiments.py --datasets clinc150

# Run specific experiment groups
python run_experiments.py --experiment-groups baseline_control ablation

# Use CPU (if no GPU available)
python run_experiments.py --device cpu

# Disable deep alignment training (for comparison experiments)
python run_experiments.py --no-deep-alignment

# Disable hybrid strategy (for comparison experiments)
python run_experiments.py --no-hybrid-strategy
```

### Combined Usage

```bash
# Run complete experiment (Deep Alignment Training + Hybrid Strategy)
python run_experiments.py \
    --use-deep-alignment \
    --use-hybrid-strategy \
    --datasets clinc150 20newsgroups \
    --models qwen2.5:3b qwen2.5:32b
```

---

## Experiment Groups

### Experiment Group 1: Baseline Control Group

**Experiment Purpose**:
- Observe forgetting under natural conditions and establish baseline
- Evaluate performance degradation of standard continual learning methods

**Run Command**:
```bash
python run_experiments.py \
  --models qwen2.5:3b \
  --datasets clinc150 \
  --experiment-groups baseline_control
```

**Key Parameters**:
- Experiment group name: `baseline_control`
- Training strategy: Standard training, no mitigation strategy
- No frozen layers
- No deep alignment training

### Experiment Group 2: Spurious Forgetting Induced Group

**Experiment Purpose**:
- Induce spurious forgetting by freezing bottom 30% of layers
- Validate characteristics of spurious forgetting: small representation space changes but large performance degradation
- Validate that reversibility score should be high (>0.6)

**Run Command**:
```bash
python run_experiments.py \
  --models qwen2.5:3b \
  --datasets clinc150 \
  --experiment-groups spurious_forgetting_induced
```

**Key Parameters**:
- Experiment group name: `spurious_forgetting_induced`
- Freezing strategy: Freeze bottom 30% of layers
- Expected results: Alignment degradation mainly occurs in top layers, high reversibility score

### Experiment Group 3: True Forgetting Induced Group

**Experiment Purpose**:
- Induce true forgetting through high-intensity training of new tasks (10 epochs), minimizing old task data
- Validate characteristics of true forgetting: large representation space changes and large performance degradation
- Validate that reversibility score should be low (<0.6)

**Run Command**:
```bash
python run_experiments.py \
  --models qwen2.5:3b \
  --datasets clinc150 \
  --experiment-groups true_forgetting_induced
```

**Key Parameters**:
- Experiment group name: `true_forgetting_induced`
- Training strategy: High-intensity training (10 epochs)
- Expected results: Fundamental changes in representation space, low reversibility score

### Experiment Group 4: Mixed Forgetting Group

**Experiment Purpose**:
- Some tasks induce spurious forgetting, some tasks induce true forgetting
- Validate identification method performance in complex scenarios
- Validate ability to distinguish different types of forgetting

**Run Command**:
```bash
python run_experiments.py \
  --models qwen2.5:3b \
  --datasets clinc150 \
  --experiment-groups mixed_forgetting
```

**Key Parameters**:
- Experiment group name: `mixed_forgetting`
- Training strategy: Combination of freezing and high-intensity training
- Expected results: Identification accuracy > 85%, able to correctly distinguish spurious and true forgetting

### Experiment Group 5: Deep Alignment Training Validation

**Experiment Purpose**:
- Validate the effectiveness of Deep Alignment Training
- Validate that deep alignment training can mitigate spurious forgetting

**Run Command**:
```bash
# Enable deep alignment training (enabled by default)
python run_experiments.py \
  --models qwen2.5:3b \
  --datasets clinc150 \
  --experiment-groups deep_alignment_training \
  --use-deep-alignment
```

**Key Parameters**:
- Experiment group name: `deep_alignment_training`
- Deep alignment threshold: τ_deep = 0.7
- Deep alignment depth: D > 10

### Experiment Group 6: Ablation Study

**Experiment Purpose**: Analyze the contribution of each component and validate the rationality of method design

**Run Command**:
```bash
# Run ablation study with single model and single dataset
python run_experiments.py \
  --models qwen2.5:3b \
  --datasets clinc150 \
  --experiment-groups ablation
```

**Ablation Components**:
1. **Full Method**: All components used
2. **No Alignment Metric**: Remove alignment metric, use only reversibility
3. **No Reversibility Metric**: Remove reversibility metric, use only alignment
4. **No Dynamic Tracking**: Do not track representation space, use only final state
5. **Fixed Strategy**: Use fixed 30% freezing, no adaptive strategy
6. **Alignment Only**: Use only alignment metric
7. **Reversibility Only**: Use only reversibility metric

### Experiment Group List

| Experiment Group Name | Description |
|----------------------|-------------|
| `baseline_control` | Baseline control group (standard training) |
| `spurious_forgetting_induced` | Spurious forgetting induced group (freeze bottom 30%) |
| `true_forgetting_induced` | True forgetting induced group (high-intensity training) |
| `mixed_forgetting` | Mixed forgetting group |
| `ablation` | Ablation study group |
| `deep_alignment_training` | Deep alignment training validation group |

---

## Experimental Results

### Result File Location

Experimental results are saved in the `./results/` directory, including:

1. **Individual Experiment Results**: `{model}_{dataset}_{group}_{timestamp}.json`
2. **Summary Report**: `summary_{timestamp}.json`

### Result Format

Each result file contains:

```json
{
  "model": "qwen2.5:3b",
  "dataset": "clinc150",
  "experiment_group": "baseline_control",
  "task_results": {
    "0": {
      "performance_before": 0.85,
      "performance_after": 0.82,
      "alignment_depth": 3,
      "reversibility_score": 0.75,
      "spurious_forgetting_score": 0.65,
      "forgetting_type": "spurious"
    }
  },
  "identification_accuracy": {
    "spurious": 0.89,
    "true": 0.87,
    "overall": 0.88
  },
  "metrics": {
    "average_accuracy": 0.764,
    "backward_transfer": -0.01
  }
}
```

### Expected Results

1. **Identification Accuracy**: Should reach 86.2%-90.6%
2. **Alignment Depth**:
   - Standard training: D ≤ 3
   - Deep alignment training: D > 12
3. **Robustness Improvement**: Deep alignment training should improve by 3.3%-7.1%

---

## Common Issues

### 1. Ollama Connection Failed

**Problem**: Unable to connect to Ollama service

**Solution**:
```bash
# Ensure Ollama service is running
ollama serve

# Check service status
curl http://localhost:11434/api/tags
```

### 2. Model Not Found

**Problem**: Model not found

**Solution**:
```bash
# Pull missing models
ollama pull qwen3:1.7b
ollama pull qwen2.5:3b
ollama pull qwen3:4b
ollama pull qwen2.5:32b

# View installed models
ollama list
```

### 3. Insufficient Memory

**Problem**: Insufficient memory when running 32B model

**Solution**:
- Use smaller models: `--models qwen2.5:3b`
- Reduce batch_size (set in config.py)
- Use CPU mode: `--device cpu` (slower)

### 4. CUDA Not Available

**Problem**: CUDA not available

**Common Causes and Solutions**:

1. **PyTorch installed is CPU version (most common)**
   ```bash
   # Uninstall CPU version
   pip uninstall torch torchvision torchaudio
   
   # Install CUDA version (choose based on your CUDA version)
   # CUDA 11.8:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   
   # CUDA 12.1:
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   
   # View all versions: https://pytorch.org/get-started/locally/
   ```

2. **NVIDIA Driver Not Installed**
   - Download and install NVIDIA driver: https://www.nvidia.com/Download/index.aspx
   - Check driver: `nvidia-smi`

3. **No NVIDIA GPU**
   - Use CPU mode: `python run_experiments.py --device cpu`
   - Note: CPU mode runs slower but works normally

### 5. Experiment Takes Too Long

**Problem**: Complete experiment takes a long time

**Solution**:
1. Run only part of models: `--models qwen2.5:3b`
2. Run only part of datasets: `--datasets clinc150`
3. Reduce number of tasks (set `num_tasks` in config.py)

---

## Project Structure

```
灾难遗忘/
├── README_CN.md                 # Project documentation (Chinese)
├── README_EN.md                 # Project documentation (English, this file)
├── requirements.txt            # Dependencies
├── config.py                   # Configuration file
├── main.py                     # Main program entry (legacy interface)
├── run_experiments.py          # Experiment main entry ⭐
├── setup_hf_mirror.py          # HuggingFace mirror setup tool
├── data/                       # Data processing module
│   ├── __init__.py
│   ├── datasets.py            # Dataset loading
│   └── clinc150_cache/        # CLINC-150 dataset cache
├── models/                     # Model definitions
│   ├── __init__.py
│   ├── base_model.py          # Base model class
│   └── qwen_model.py          # Qwen model (local and Ollama, used in experiments) ⭐
├── metrics/                    # Evaluation metrics
│   ├── __init__.py
│   ├── alignment_score.py     # Alignment metric
│   ├── reversibility.py       # Reversibility metric
│   ├── forgetting_metrics.py  # Forgetting metrics
│   └── evaluation.py          # Comprehensive evaluation
├── analysis/                   # Representation space analysis tools
│   ├── __init__.py
│   ├── alignment_analyzer.py  # Alignment analysis tool
│   ├── reversibility_analyzer.py # Reversibility analysis tool
│   ├── representation_tracker.py # Dynamic tracking tool
│   ├── cka_implementation.py  # CKA (Centered Kernel Alignment) implementation
│   └── visualization.py       # Visualization tool
├── experiments/                # Experiment scripts
│   ├── __init__.py
│   ├── main_experiment.py     # Main experiment script ⭐
│   └── spurious_forgetting_identification.py  # Spurious forgetting identification experiment
├── training/                   # Training module
│   ├── __init__.py
│   ├── deep_alignment_trainer.py # Deep alignment training ⭐
│   └── adaptive_mitigation.py  # Adaptive mitigation strategy ⭐
├── utils/                      # Utility functions
│   ├── __init__.py
│   ├── logger.py              # Logging tool
│   ├── helpers.py             # Helper functions
│   └── ollama_helper.py       # Ollama helper tool
├── results/                    # Experimental results directory (automatically generated)
├── logs/                       # Log directory
└── examples/                   # Example code directory
```

---

## Datasets

This project uses the following open-source datasets:

1. **CLINC-150**: Natural language understanding dataset, 15 intent classification tasks
2. **20 Newsgroups**: Text classification dataset, 5 tasks

---

## Supported Models

This experiment uses the following Qwen models (deployed via Ollama):

1. **Qwen3-1.7B** (`qwen3:1.7b`) - 1.7B parameters
2. **Qwen2.5-3B** (`qwen2.5:3b`) - 3B parameters
3. **Qwen3-4B** (`qwen3:4b`) - 4B parameters
4. **Qwen2.5-32B** (`qwen2.5:32b`) - 32B parameters ⭐ **Recommended**

**Advantages of Using Ollama Models**:
- Support for larger models (e.g., 32B)
- No need to download full models locally
- Automatic GPU acceleration (if available)
- More flexible model management

---

## Evaluation Metrics

- **Identification Accuracy**: Identification accuracy for spurious and true forgetting
- **Alignment Metric**: Quantitative measure of task alignment
- **Reversibility Metric**: Score for forgetting reversibility
- **Average Accuracy**: Average accuracy across all tasks
- **Forgetting Rate**: Performance degradation rate of old tasks
- **Forward/Backward Transfer**: Knowledge transfer effects

---

## Command Line Arguments

### Common Parameter Descriptions

| Parameter | Description | Default Value | Example |
|-----------|------------|---------------|---------|
| `--models` | List of models to use | All 4 models | `--models qwen2.5:3b` |
| `--datasets` | Datasets to use | `clinc150 20newsgroups` | `--datasets clinc150` |
| `--experiment-groups` | Experiment groups to run | All groups | `--experiment-groups baseline_control` |
| `--device` | Computing device | `cuda` | `--device cpu` |
| `--use-deep-alignment` | Enable deep alignment training | `True` | `--use-deep-alignment` |
| `--no-deep-alignment` | Disable deep alignment training | - | `--no-deep-alignment` |
| `--use-hybrid-strategy` | Enable hybrid mitigation strategy | `True` | `--use-hybrid-strategy` |
| `--no-hybrid-strategy` | Disable hybrid mitigation strategy | - | `--no-hybrid-strategy` |
| `--ollama-url` | Ollama service address | `http://localhost:11434` | `--ollama-url http://localhost:11434` |
| `--check-ollama` | Only check Ollama setup | - | `--check-ollama` |

### View Help

```bash
python run_experiments.py --help
```

---

## Recommended Workflow

### First Run

```bash
# 1. Check environment
python run_experiments.py --check-ollama

# 2. Run small-scale test (using small model)
python run_experiments.py --models qwen2.5:3b --datasets clinc150

# 3. If test succeeds, run complete experiment
python run_experiments.py
```

### Daily Run

```bash
# Direct run experiments
python run_experiments.py
```

### Tips

- For first run, recommend testing with small models (e.g., `qwen2.5:3b`)
- Complete experiments may take a long time, recommend running in background
- Results are automatically saved, can interrupt and resume at any time
- If encountering issues, first run `--check-ollama` to check environment

