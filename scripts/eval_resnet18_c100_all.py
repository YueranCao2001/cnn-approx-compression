import os
import time
import torch
import torch.nn as nn
from torchvision.models import resnet18

from train_resnet18_c100_baseline import get_loaders_c100, evaluate


def get_file_size(path):
    """Return file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def load_fp32_model(path, num_classes=100):
    """
    Load a standard FP32 ResNet-18 model using its state_dict.
    Used for the baseline and pruned models on CIFAR-100.
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
    on a randomly generated CIFAR-sized input.
    Returns average seconds per image.
    """
    model.to(device)
    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)

    with torch.no_grad():
        for _ in range(10):
            _ = model(x)

        start = time.time()
        for _ in range(runs):
            _ = model(x)
        end = time.time()

    return (end - start) / runs


def collect_metrics_c100():
    """
    Evaluate baseline and 50% pruned ResNet-18 on CIFAR-100 (CPU).

    Returns a dict:
        {
            "names": [...],
            "accuracy": [...],    # in [0, 1]
            "size_mb": [...],     # checkpoint size in MB
            "latency_s": [...],   # seconds per image (CPU)
        }
    """
    device = "cpu"
    _, testloader = get_loaders_c100()

    base_path = "models/resnet18_c100_base.pth"
    pruned_path = "models/resnet18_c100_pruned50.pth"

    size_base = get_file_size(base_path)
    size_pruned = get_file_size(pruned_path)

    base = load_fp32_model(base_path)
    pruned = load_fp32_model(pruned_path)

    acc_base = evaluate(base.to(device), testloader, device)
    acc_pruned = evaluate(pruned.to(device), testloader, device)

    t_base = speed_test(base, device=device)
    t_pruned = speed_test(pruned, device=device)

    metrics = {
        "names": ["Baseline FP32", "Pruned 50% FP32"],
        "accuracy": [acc_base, acc_pruned],
        "size_mb": [size_base, size_pruned],
        "latency_s": [t_base, t_pruned],
    }
    return metrics


def main():
    metrics = collect_metrics_c100()

    names = metrics["names"]
    acc = metrics["accuracy"]
    size_mb = metrics["size_mb"]
    latency_s = metrics["latency_s"]

    print("[CIFAR-100] File sizes (MB):")
    for name, s in zip(names, size_mb):
        print(f"  {name:18s}: {s:.4f} MB")

    print("\n[CIFAR-100] Accuracy (CPU):")
    for name, a in zip(names, acc):
        print(f"  {name:18s}: {a:.4f}")

    print("\n[CIFAR-100] Average inference time per image (CPU):")
    for name, t in zip(names, latency_s):
        print(f"  {name:18s}: {t * 1000:.3f} ms")


if __name__ == "__main__":
    main()
