import os
import csv

import torch
import torch.nn as nn
from torchvision.models import resnet18

# Output directory for CSV results
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)


def build_resnet18_c10(num_classes=10):
    """
    Construct a ResNet-18 model configured for CIFAR-10 classification.
    The architecture matches the model used when saving the checkpoints.
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def compute_layer_sparsity(model):
    """
    Compute sparsity statistics for each Conv2d / Linear layer.
    Sparsity is defined as the percentage of weights that are zero.

    Returns:
        A list of dictionaries, each containing:
        - layer_name: module name
        - weight_shape: shape of the weight tensor
        - total_params: total number of weights
        - zero_params: number of zero-valued weights
        - sparsity: zero_params / total_params
    """
    stats = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            w = module.weight.data
            total = w.numel()
            zeros = (w == 0).sum().item()
            sparsity = zeros / total if total > 0 else 0.0

            stats.append(
                {
                    "layer_name": name,
                    "weight_shape": list(w.shape),
                    "total_params": total,
                    "zero_params": zeros,
                    "sparsity": sparsity,
                }
            )

    return stats


def save_stats_to_csv(stats, csv_path):
    """
    Save sparsity statistics into a CSV file.

    Columns:
        layer_name, weight_shape, total_params, zero_params, sparsity
    """
    fieldnames = [
        "layer_name",
        "weight_shape",
        "total_params",
        "zero_params",
        "sparsity",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in stats:
            row_out = row.copy()
            # Convert shape list to a readable string for CSV
            row_out["weight_shape"] = str(row_out["weight_shape"])
            writer.writerow(row_out)

    print(f"Saved layer sparsity to {csv_path}")


def summarize_model_sparsity(stats):
    """
    Summarize model-wide sparsity by aggregating over all Conv/Linear layers.

    Returns:
        (total_params, total_zero_params, sparsity_percentage)
    """
    total_params = sum(r["total_params"] for r in stats)
    zero_params = sum(r["zero_params"] for r in stats)
    sparsity = zero_params / total_params if total_params > 0 else 0.0
    return total_params, zero_params, sparsity


def analyze_checkpoint(label, ckpt_path, csv_suffix):
    """
    Load a checkpoint, compute sparsity, print summary,
    and save layer-wise sparsity into a CSV.

    Args:
        label: descriptive name for this analysis run
        ckpt_path: model checkpoint (.pth)
        csv_suffix: output CSV filename under results/
    """
    if not os.path.exists(ckpt_path):
        print(f"[WARN] Checkpoint not found: {ckpt_path}, skipping.")
        return

    print(f"\n===== Analyzing {label} =====")
    print(f"Loading checkpoint: {ckpt_path}")

    # Load model
    model = build_resnet18_c10()
    state_dict = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state_dict)

    # Compute layer-wise sparsity
    stats = compute_layer_sparsity(model)

    # Print first few layers for inspection
    print("Example layers (first 5):")
    for row in stats[:5]:
        print(
            f"  {row['layer_name']:25s} | "
            f"shape={row['weight_shape']} | "
            f"sparsity={row['sparsity']:.4f}"
        )

    # Compute global sparsity summary
    total_params, total_zeros, sparsity = summarize_model_sparsity(stats)
    print(
        f"Total parameters = {total_params}, "
        f"zeros = {total_zeros}, "
        f"global sparsity = {sparsity:.4f} ({sparsity * 100:.2f}%)"
    )

    # Save to CSV
    csv_path = os.path.join(RESULT_DIR, csv_suffix)
    save_stats_to_csv(stats, csv_path)


def main():
    """
    Analyze ResNet-18 checkpoints on CIFAR-10 for different pruning levels:
        - Baseline (0% pruning)
        - 30% pruning
        - 50% pruning
        - 70% pruning

    For each model, a layer-wise sparsity CSV is generated.
    """
    checkpoints = [
        ("ResNet18 C10 Base",   "models/resnet18_c10_base.pth",   "resnet18_c10_base_sparsity.csv"),
        ("ResNet18 C10 P30",    "models/resnet18_c10_pruned30.pth", "resnet18_c10_pruned30_sparsity.csv"),
        ("ResNet18 C10 P50",    "models/resnet18_c10_pruned50.pth", "resnet18_c10_pruned50_sparsity.csv"),
        ("ResNet18 C10 P70",    "models/resnet18_c10_pruned70.pth", "resnet18_c10_pruned70_sparsity.csv"),
    ]

    for label, ckpt, csvname in checkpoints:
        analyze_checkpoint(label, ckpt, csvname)

    print("\nDone. All sparsity CSVs saved under 'results/'.")


if __name__ == "__main__":
    main()
