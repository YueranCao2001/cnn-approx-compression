import os
import time
import torch
import torch.nn as nn
from torchvision.models import resnet18

from train_resnet18_c10_baseline import get_loaders, evaluate


def get_file_size(path):
    return os.path.getsize(path) / (1024 * 1024)


def load_fp32_model(path, num_classes=10):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
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


def collect_prune_sweep_metrics():
    """
    Evaluate ResNet-18 on CIFAR-10 for pruning ratios: 0, 0.3, 0.5, 0.7.
    Returns:
        {
          "ratios": [0.0, 0.3, 0.5, 0.7],
          "names":  [...],
          "accuracy": [...],
          "size_mb": [...],
          "latency_s": [...],
        }
    """
    device = "cpu"
    _, testloader = get_loaders()

    ckpts = [
        ("Baseline", 0.0, "models/resnet18_c10_base.pth"),
        ("Pruned30", 0.3, "models/resnet18_c10_pruned30.pth"),
        ("Pruned50", 0.5, "models/resnet18_c10_pruned50.pth"),
        ("Pruned70", 0.7, "models/resnet18_c10_pruned70.pth"),
    ]

    names, ratios, acc_list, size_list, lat_list = [], [], [], [], []

    for name, ratio, path in ckpts:
        if not os.path.exists(path):
            print(f"[WARN] Missing checkpoint: {path} (skip this one)")
            continue

        print(f"Loading {name} from {path}")
        model = load_fp32_model(path)

        print("  Evaluating accuracy...")
        acc = evaluate(model.to(device), testloader, device)

        print("  Measuring latency...")
        t = speed_test(model, device=device)

        s = get_file_size(path)

        names.append(name)
        ratios.append(ratio)
        acc_list.append(acc)
        size_list.append(s)
        lat_list.append(t)

        print(f"  -> acc={acc:.4f}, size={s:.4f} MB, latency={t*1000:.3f} ms\n")

    return {
        "names": names,
        "ratios": ratios,
        "accuracy": acc_list,
        "size_mb": size_list,
        "latency_s": lat_list,
    }


def main():
    metrics = collect_prune_sweep_metrics()

    print("Summary (ratio, name, acc, size_MB, latency_ms):")
    for r, n, a, s, t in zip(
        metrics["ratios"],
        metrics["names"],
        metrics["accuracy"],
        metrics["size_mb"],
        metrics["latency_s"],
    ):
        print(f"  {r:.1f}  {n:10s} | acc={a:.4f} | size={s:.4f} MB | {t*1000:.3f} ms")


if __name__ == "__main__":
    main()
