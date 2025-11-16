import os

# Fix for "OMP: Error #15: Initializing libiomp5md.dll" on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import torch
import torch.nn as nn

import matplotlib
matplotlib.use("Agg")  # use non-GUI backend so saving works on Windows
import matplotlib.pyplot as plt

from torchvision.models import resnet18
from train_baseline import get_loaders, evaluate

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)



def get_file_size(path):
    """Return file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def load_fp32_model(path, num_classes=10):
    """Load baseline / pruned FP32 ResNet-18 from a state_dict."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_int8_model(path, device="cpu"):
    """Load the quantized model saved as a full model object."""
    model = torch.load(path, map_location=device)
    model.eval()
    return model


def speed_test(model, device="cpu", runs=200):
    """Measure average inference time (seconds / image) on a dummy CIFAR-10 input."""
    model.to(device)
    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)

    with torch.no_grad():
        for _ in range(10):  # warm-up
            _ = model(x)
        start = time.time()
        for _ in range(runs):
            _ = model(x)
        end = time.time()

    return (end - start) / runs


def collect_metrics():
    """Collect accuracy, model size, and latency for the three models."""
    device = "cpu"
    _, testloader = get_loaders()

    print("Loading models and computing file sizes...")
    size_base = get_file_size("models/resnet18_base.pth")
    size_pruned = get_file_size("models/resnet18_pruned.pth")
    size_int8 = get_file_size("models/resnet18_pruned_int8.pth")

    base = load_fp32_model("models/resnet18_base.pth")
    pruned = load_fp32_model("models/resnet18_pruned.pth")
    quantized = load_int8_model("models/resnet18_pruned_int8.pth", device=device)

    print("Evaluating accuracy...")
    acc_base = evaluate(base.to(device), testloader, device)
    acc_pruned = evaluate(pruned.to(device), testloader, device)
    acc_int8 = evaluate(quantized.to(device), testloader, device)

    print("Measuring latency...")
    t_base = speed_test(base, device=device)
    t_pruned = speed_test(pruned, device=device)
    t_int8 = speed_test(quantized, device=device)

    metrics = {
        "names": ["Baseline FP32", "Pruned FP32", "Pruned + INT8"],
        "accuracy": [acc_base, acc_pruned, acc_int8],
        "size_mb": [size_base, size_pruned, size_int8],
        "latency_s": [t_base, t_pruned, t_int8],
    }
    print("Metrics:", metrics)
    return metrics


def plot_bar(values, labels, ylabel, title, filename):
    """Helper to create a bar plot and save it to results/."""
    plt.figure(figsize=(6, 4))
    x = range(len(values))
    plt.bar(x, values)
    plt.xticks(x, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_path = os.path.join(RESULT_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_scatter_accuracy_size(accuracy, size_mb, labels, filename):
    """Scatter plot of accuracy vs model size."""
    plt.figure(figsize=(6, 4))
    for acc, size, label in zip(accuracy, size_mb, labels):
        plt.scatter(size, acc)
        plt.text(size, acc, label, fontsize=9, ha="left", va="bottom")
    plt.xlabel("Model size (MB)")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Model Size")
    plt.tight_layout()
    out_path = os.path.join(RESULT_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Collecting metrics...")
    metrics = collect_metrics()

    names = metrics["names"]
    acc = metrics["accuracy"]
    size_mb = metrics["size_mb"]
    latency_s = metrics["latency_s"]

    print("Generating plots...")
    plot_bar(
        values=acc,
        labels=names,
        ylabel="Accuracy",
        title="Accuracy on CIFAR-10",
        filename="accuracy_bar.png",
    )

    plot_bar(
        values=size_mb,
        labels=names,
        ylabel="Size (MB)",
        title="Model Size",
        filename="size_bar.png",
    )

    plot_scatter_accuracy_size(
        accuracy=acc,
        size_mb=size_mb,
        labels=names,
        filename="accuracy_vs_size.png",
    )

    latency_ms = [t * 1000 for t in latency_s]
    plot_bar(
        values=latency_ms,
        labels=names,
        ylabel="Latency (ms / image, CPU)",
        title="Inference Latency (CPU)",
        filename="latency_bar.png",
    )

    print("Done! All plots saved in 'results/'.")


if __name__ == "__main__":
    main()
