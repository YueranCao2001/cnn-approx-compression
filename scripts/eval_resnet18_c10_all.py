import os
import time
import torch
import torch.nn as nn
from torchvision.models import resnet18

from train_resnet18_c10_baseline import get_loaders, evaluate


def get_file_size(path):
    """Return file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def load_fp32_model(path, num_classes=10):
    """
    Load a standard FP32 ResNet-18 model using its state_dict.
    Used for the baseline and pruned models.
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def speed_test(model, device="cpu", runs=200):
    """
    Measure inference latency by running several forward passes
    on a randomly generated CIFAR-10-sized input.
    Returns average seconds per image.
    """
    model.to(device)
    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)

    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(x)

        start = time.time()
        for _ in range(runs):
            _ = model(x)
        end = time.time()

    return (end - start) / runs  # average seconds per image


def collect_metrics():
    """
    Evaluate baseline, pruned, and pruned+INT8 models on CIFAR-10 (CPU).

    Returns a dict:
        {
            "names": [...],
            "accuracy": [...],    # in [0, 1]
            "size_mb": [...],     # checkpoint size in MB
            "latency_s": [...],   # seconds per image (CPU)
        }
    """
    device = "cpu"
    _, testloader = get_loaders()

    # Paths for the three model variants
    base_path = "models/resnet18_c10_base.pth"
    pruned_path = "models/resnet18_c10_pruned50.pth"
    int8_path = "models/resnet18_c10_pruned50_int8.pth"

    # File sizes
    size_base = get_file_size(base_path)
    size_pruned = get_file_size(pruned_path)
    size_int8 = get_file_size(int8_path)

    # Load FP32 models
    base = load_fp32_model(base_path)
    pruned = load_fp32_model(pruned_path)

    # Load INT8 model (full model, not state_dict)
    quantized = torch.load(int8_path, map_location=device)
    quantized.eval()

    # Accuracy
    acc_base = evaluate(base.to(device), testloader, device)
    acc_pruned = evaluate(pruned.to(device), testloader, device)
    acc_quant = evaluate(quantized.to(device), testloader, device)

    # Latency (seconds per image)
    t_base = speed_test(base, device)
    t_pruned = speed_test(pruned, device)
    t_quant = speed_test(quantized, device)

    metrics = {
        "names": ["Baseline FP32", "Pruned FP32", "Pruned + INT8"],
        "accuracy": [acc_base, acc_pruned, acc_quant],
        "size_mb": [size_base, size_pruned, size_int8],
        "latency_s": [t_base, t_pruned, t_quant],
    }
    return metrics


def main():
    metrics = collect_metrics()

    names = metrics["names"]
    acc = metrics["accuracy"]
    size_mb = metrics["size_mb"]
    latency_s = metrics["latency_s"]

    print("File sizes (MB):")
    for name, s in zip(names, size_mb):
        print(f"  {name:18s}: {s:.4f} MB")

    print("\nAccuracy on CIFAR-10 (CPU):")
    for name, a in zip(names, acc):
        print(f"  {name:18s}: {a:.4f}")

    print("\nAverage inference time per image (CPU):")
    for name, t in zip(names, latency_s):
        print(f"  {name:18s}: {t * 1000:.3f} ms")


if __name__ == "__main__":
    main()
