import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_resnet18_c10_prune_sweep import collect_prune_sweep_metrics

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

PREFIX = "resnet18_c10_prunesweep"

FOOTER = (
    "Model: ResNet-18 · Dataset: CIFAR-10 · "
    "Global unstructured L1 pruning on Conv + Linear · "
    "Ratios: 0 / 30% / 50% / 70%"
)


def plot_line(x, y, xlabel, ylabel, title, filename, value_format=None, ylim=None):
    plt.figure(figsize=(7, 4))
    plt.plot(x, y, marker="o")

    for xx, yy in zip(x, y):
        if value_format:
            txt = value_format.format(yy)
        else:
            txt = f"{yy:.3f}"
        plt.text(xx, yy, txt, fontsize=9, ha="center", va="bottom")

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.figtext(0.99, 0.01, FOOTER, ha="right", va="bottom", fontsize=8)

    plt.tight_layout(rect=[0.04, 0.08, 0.97, 0.95])
    out_path = os.path.join(RESULT_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    metrics = collect_prune_sweep_metrics()

    ratios = metrics["ratios"]          # [0.0, 0.3, 0.5, 0.7]
    acc = [a * 100.0 for a in metrics["accuracy"]]   # 转百分比
    size_mb = metrics["size_mb"]
    latency_ms = [t * 1000.0 for t in metrics["latency_s"]]

    # 为了让变化更明显，稍微缩一下 y 轴范围
    if acc:
        acc_ylim = (min(acc) - 1.0, max(acc) + 1.0)
    else:
        acc_ylim = None

    if size_mb:
        size_ylim = (min(size_mb) - 0.5, max(size_mb) + 0.5)
    else:
        size_ylim = None

    if latency_ms:
        lat_ylim = (min(latency_ms) - 0.3, max(latency_ms) + 0.3)
    else:
        lat_ylim = None

    # 1) Accuracy vs pruning ratio
    plot_line(
        x=ratios,
        y=acc,
        xlabel="Pruning ratio (fraction of weights removed)",
        ylabel="Accuracy (%)",
        title="Accuracy vs Pruning Ratio\nResNet-18 · CIFAR-10",
        filename=f"{PREFIX}_accuracy_vs_ratio.png",
        value_format="{:.1f}%",
        ylim=acc_ylim,
    )

    # 2) Model size vs pruning ratio
    plot_line(
        x=ratios,
        y=size_mb,
        xlabel="Pruning ratio",
        ylabel="Model size (MB)",
        title="Model Size vs Pruning Ratio\nResNet-18 · CIFAR-10",
        filename=f"{PREFIX}_modelsize_vs_ratio.png",
        value_format="{:.5f} MB",
        ylim=size_ylim,
    )

    # 3) Latency vs pruning ratio
    plot_line(
        x=ratios,
        y=latency_ms,
        xlabel="Pruning ratio",
        ylabel="Latency (ms / image, CPU)",
        title="Inference Latency vs Pruning Ratio (CPU)\nResNet-18 · CIFAR-10",
        filename=f"{PREFIX}_latency_vs_ratio.png",
        value_format="{:.3f} ms",
        ylim=lat_ylim,
    )

    print("Done. Pruning-sweep plots saved in 'results/'.")


if __name__ == "__main__":
    main()
