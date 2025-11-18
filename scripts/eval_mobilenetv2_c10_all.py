import os
import time
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2

from train_resnet18_c10_baseline import get_loaders, evaluate


def get_file_size(path):
    return os.path.getsize(path) / (1024 * 1024)


def load_mobilenet_c10(path, num_classes=10):
    model = mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def speed_test(model, device="cpu", runs=200):
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


def collect_metrics_mnet_c10():
    """
    Evaluate MobileNetV2 (baseline & 50% pruned) on CIFAR-10 (CPU).
    Returns:
        {
          "names": [...],
          "accuracy": [...],
          "size_mb": [...],
          "latency_s": [...],
        }
    """
    device = "cpu"
    _, testloader = get_loaders()

    base_path = "models/mobilenetv2_c10_base.pth"
    pruned_path = "models/mobilenetv2_c10_pruned50.pth"

    size_base = get_file_size(base_path)
    size_pruned = get_file_size(pruned_path)

    base = load_mobilenet_c10(base_path)
    pruned = load_mobilenet_c10(pruned_path)

    acc_base = evaluate(base.to(device), testloader, device)
    acc_pruned = evaluate(pruned.to(device), testloader, device)

    t_base = speed_test(base, device=device)
    t_pruned = speed_test(pruned, device=device)

    metrics = {
        "names": ["MobileNetV2 FP32", "MobileNetV2 Pruned 50%"],
        "accuracy": [acc_base, acc_pruned],
        "size_mb": [size_base, size_pruned],
        "latency_s": [t_base, t_pruned],
    }
    return metrics


def main():
    metrics = collect_metrics_mnet_c10()

    names = metrics["names"]
    acc = metrics["accuracy"]
    size_mb = metrics["size_mb"]
    latency_s = metrics["latency_s"]

    print("[CIFAR-10][MobileNetV2] File sizes (MB):")
    for name, s in zip(names, size_mb):
        print(f"  {name:24s}: {s:.4f} MB")

    print("\n[CIFAR-10][MobileNetV2] Accuracy (CPU):")
    for name, a in zip(names, acc):
        print(f"  {name:24s}: {a:.4f}")

    print("\n[CIFAR-10][MobileNetV2] Average inference time per image (CPU):")
    for name, t in zip(names, latency_s):
        print(f"  {name:24s}: {t * 1000:.3f} ms")


if __name__ == "__main__":
    main()
