# Approximate Computing for CNN Compression

This repository implements a small experimental framework to study **approximate computing techniques** for CNN compression, inspired by the *Deep Compression* paper (Han et al., ICLR 2016).  
We compare three model variants:

1. **Baseline FP32 model** (no approximation)  
2. **Pruned FP32 model** (global unstructured pruning + fine-tuning)  
3. **Pruned + INT8 model** (pruning + dynamic quantization)

Experiments are run on **ResNet-18** trained on **CIFAR-10** using PyTorch.

---

## 1. Environment Setup

We recommend using Anaconda.

# Create and activate a new environment
```conda create -n approx_cnn python=3.10 -y```

```conda activate approx_cnn```

# Install PyTorch (CPU-only example; replace with CUDA version if you have a GPU)
```conda install -y pytorch torchvision torchaudio cpuonly -c pytorch -c conda-forge```

# Extra packages
```conda install -y matplotlib```

```pip install tqdm```

# Clone or place this repo in your working directory, then move into it:
```cd cnn-approx-compression```

## 2. Project Structure

```cnn-approx-compression/```

```├── data/```                     # CIFAR-10 will be downloaded here automatically

```├── models/```                   # Saved model checkpoints (.pth)

```├── results/```                  # Plots (accuracy, size, latency)

```├── scripts/```

```|      ├── train_baseline.py```     # Train baseline ResNet-18 on CIFAR-10

```|      ├── prune_model.py```        # Apply global pruning + fine-tuning

```|      ├── quantize_model.py```     # Apply dynamic INT8 quantization

```|      ├── eval_model.py```         # Print accuracy, size, and latency

```|      └── plot_results.py```       # Generate result plots into results/

```└── README.md```


## 3. Pipeline Overview
The full experimental pipeline is:
1. Train baseline ResNet-18 (FP32)
2. Apply global unstructured pruning + fine-tuning
3. Apply dynamic INT8 quantization to the pruned model
4. Evaluate accuracy, model size, and inference latency
5. Visualize the trade-offs using plots

Each step corresponds to one script in scripts/

# 4. Running the Experiments
## 4.1 Train the Baseline Model
### Purpose:
Train a standard FP32 ResNet-18 on CIFAR-10. This model serves as the baseline for all comparisons.

```conda activate approx_cnn```

```python scripts/train_baseline.py```

### What it does:
Downloads CIFAR-10 into data/ (if not already present)

Trains ResNet-18 for a fixed number of epochs

Tracks test accuracy and saves the best model to: ```models/resnet18_base.pth```

This checkpoint is the starting point for pruning and quantization.

## 4.2 Prune the Baseline Model
### Purpose:
Apply global unstructured pruning to reduce the number of non-zero weights, then fine-tune to recover accuracy.
```python scripts/prune_model.py```

### What it does:

Loads ```models/resnet18_base.pth```

Applies global L1-unstructured pruning to all Conv2d and Linear layers

Fine-tunes the pruned model for a few epochs

Removes pruning reparameterization and saves the final pruned weights to: ```models/resnet18_pruned.pth```

### Notes:
The state_dict is still stored in FP32 format, so the file size on disk does not change, but the effective number of non-zero weights is reduced.

## 4.3 Quantize the Pruned Model (INT8)
### Purpose:
Apply dynamic quantization to convert Linear layers from FP32 to INT8, on top of the pruned model.

```python scripts/quantize_model.py```

### What it does:
Loads ```models/resnet18_pruned.pth```

Uses torch.quantization.quantize_dynamic to quantize nn.Linear layers to INT8

Evaluates the pruned FP32 and pruned+INT8 models on CIFAR-10

Saves the full quantized model object to: ```models/resnet18_pruned_int8.pth```

### Now we have three model variants:
```resnet18_base.pth``` – Baseline FP32

```resnet18_pruned.pth``` – Pruned FP32

```resnet18_pruned_int8.pth``` – Pruned + INT8 quantized


## 4.4 Evaluate Accuracy, Size, and Latency

### Purpose:
Compare the three models under common metrics.
```python scripts/eval_model.py```

### What it prints:
Accuracy on CIFAR-10 (CPU) for each model

File size (MB) of each checkpoint

Average inference time per image (ms, CPU) using a dummy input


## 4.5 Generate Plots
### Purpose:
Create visual summaries of the experimental results.

```python scripts/plot_results.py```

### What it does:
Collects metrics (accuracy, size, latency) for the three models

Generates the following plots in the results/ folder:

```accuracy_bar.png – accuracy comparison (bar chart)```

```size_bar.png – model size comparison (bar chart)```

```accuracy_vs_size.png – accuracy vs model size (scatter plot)```

```latency_bar.png – CPU latency comparison (bar chart)```


# 5. What “Baseline”, “Pruned”, and “Pruned + INT8” Mean

## Baseline FP32
A standard ResNet-18 trained on CIFAR-10 with 32-bit floating-point weights.

Represents the uncompressed / non-approximate model.

Used as the reference point for accuracy, storage, and latency.


## Pruned FP32
Same architecture as the baseline, but many weights are set to zero via global unstructured pruning.

Fine-tuning is applied to recover accuracy after pruning.

Reduces the number of effective parameters, but in this implementation the FP32 state_dict is still stored densely, so on-disk size does not shrink.


## Pruned + INT8
Starts from the pruned model and applies dynamic INT8 quantization to Linear layers.

Further reduces arithmetic precision for part of the model.

In our experiments, this typically causes only a small drop in accuracy, if any.


Together, these three variants allow us to study the trade-off between accuracy, model complexity, and runtime cost, which is precisely the goal described in our project proposal and mid-term report.

# 6. Notes and Limitations

The current implementation focuses on unstructured pruning and dynamic quantization in PyTorch.

We do not implement Huffman coding or sparse storage formats, so compressed on-disk size is not reduced as aggressively as in the original Deep Compression paper.

All evaluation scripts run on CPU by default for reproducibility and easier comparison.

