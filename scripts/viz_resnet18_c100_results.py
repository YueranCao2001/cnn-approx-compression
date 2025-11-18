import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_resnet18_c100_all import collect_metrics_c100


RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

PREFIX = "resnet18_c100_prune50_fp32"

EXPERIMENT_FOOTER = (
    "Model: ResNet-18 · Dataset: CIFAR-100 · "
    "Pruning: 50% global L1 (Conv + Linear) · "
    "No quantization (FP32 only)"
)


def plot_bar(values, labels, ylabel, title, filename,
             value_format=None, ylim=None):
    plt.figure(figsize=(7, 4))
    x = range(len(values))
    bars = plt.bar(x, values)

    plt.xticks(x, labels, rotation=15)
    plt.ylabel(ylabel)
    plt.title(title)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.grid(axis="y", linestyle="--", alpha=0.4)

    for i, bar in enumerate(bars):
        v = values[i]
        if value_format is not None:
            txt = value_format.format(v)
        else:
            txt = f"{v:.3f}"
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height(),
            txt,
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.figtext(
        0.99, 0.01, EXPERIMENT_FOOTER,
        ha="right", va="bottom", fontsize=8
    )

    plt.tight_layout(rect=[0.03, 0.08, 0.97, 0.95])
    out_path = os.path.join(RESULT_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def plot_scatter_accuracy_size(accuracy_pct, size_mb, labels, filename):
    plt.figure(figsize=(7, 4))

    for acc, size, label in zip(accuracy_pct, size_mb, labels):
        plt.scatter(size, acc, s=60, marker="o")
        plt.text(size + 0.0005, acc, label, fontsize=9, va="center")

    xmin = min(size_mb) - 0.003
    xmax = max(size_mb) + 0.003
    plt.xlim(xmin, xmax)

    ymin = min(accuracy_pct) - 2.0
    ymax = max(accuracy_pct) + 2.0
    plt.ylim(ymin, ymax)

    plt.xlabel("Model size (MB)")
    plt.ylabel("Accuracy (%)")
    plt.title("Accuracy vs Model Size\nResNet-18 · CIFAR-100 · 50% pruning (FP32)")
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.figtext(
        0.99, 0.01, EXPERIMENT_FOOTER,
        ha="right", va="bottom", fontsize=8
    )

    plt.tight_layout(rect=[0.03, 0.08, 0.97, 0.95])
    out_path = os.path.join(RESULT_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Collecting CIFAR-100 metrics ...")
    metrics = collect_metrics_c100()

    raw_names = metrics["names"]          # ['Baseline FP32', 'Pruned 50% FP32']
    display_names = ["Base FP32", "Pruned 50% FP32"]

    acc_raw = metrics["accuracy"]
    size_mb = metrics["size_mb"]
    latency_s = metrics["latency_s"]

    acc_pct = [a * 100.0 for a in acc_raw]
    acc_min = min(acc_pct) - 1.0
    acc_max = max(acc_pct) + 1.0

    size_min = min(size_mb) - 0.02
    size_max = max(size_mb) + 0.02

    latency_ms = [t * 1000.0 for t in latency_s]
    lat_min = min(latency_ms) - 0.2
    lat_max = max(latency_ms) + 0.2

    print("Generating CIFAR-100 plots...")

    plot_bar(
        values=acc_pct,
        labels=display_names,
        ylabel="Accuracy (%)",
        title="Accuracy vs Model Variant\nResNet-18 · CIFAR-100 · 50% pruning (FP32)",
        filename=f"{PREFIX}_accuracy.png",
        value_format="{:.1f}%",
        ylim=(acc_min, acc_max),
    )

    plot_bar(
        values=size_mb,
        labels=display_names,
        ylabel="Size (MB)",
        title="Model Size vs Variant\nResNet-18 · CIFAR-100 · 50% pruning (FP32)",
        filename=f"{PREFIX}_modelsize.png",
        value_format="{:.4f} MB",
        ylim=(size_min, size_max),
    )

    plot_scatter_accuracy_size(
        accuracy_pct=acc_pct,
        size_mb=size_mb,
        labels=display_names,
        filename=f"{PREFIX}_acc_vs_size.png",
    )

    plot_bar(
        values=latency_ms,
        labels=display_names,
        ylabel="Latency (ms / image, CPU)",
        title="Inference Latency vs Variant (CPU)\nResNet-18 · CIFAR-100 · 50% pruning (FP32)",
        filename=f"{PREFIX}_latency.png",
        value_format="{:.3f} ms",
        ylim=(lat_min, lat_max),
    )

    print("Done! CIFAR-100 plots saved in 'results/'.")


if __name__ == "__main__":
    main()
