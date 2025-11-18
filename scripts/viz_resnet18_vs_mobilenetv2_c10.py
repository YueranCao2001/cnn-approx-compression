import os

# Windows libiomp
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from eval_resnet18_c10_all import collect_metrics as collect_resnet_metrics
from eval_mobilenetv2_c10_all import collect_metrics_mnet_c10


RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

PREFIX = "resnet18_vs_mnetv2_c10_prune50"

FOOTER = (
    "Dataset: CIFAR-10 · "
    "Pruning: 50% global L1 (Conv + Linear) · "
    "FP32 inference on CPU"
)


def grouped_bar(
    x_labels,                           # e.g. ["ResNet-18", "MobileNetV2"]
    group_values,                       # shape: [num_variants][num_models]
    variant_labels,                     # e.g. ["Base", "Pruned 50%"]
    ylabel,
    title,
    filename,
    value_format=None,
    ylim=None,
):
    """
     grouped bar:
      x (ResNet / MobileNet)
       (Base / Pruned)
    """
    num_models = len(x_labels)
    num_variants = len(variant_labels)

    plt.figure(figsize=(7, 4))

    x = range(num_models)
    total_width = 0.7
    bar_width = total_width / num_variants

    offsets = [
        -total_width / 2 + (i + 0.5) * bar_width for i in range(num_variants)
    ]

    for vidx, (offset, variant_name, vals) in enumerate(
        zip(offsets, variant_labels, group_values)
    ):
        xs = [xx + offset for xx in x]
        bars = plt.bar(xs, vals, width=bar_width, label=variant_name)

        for bx, by in zip(xs, vals):
            if value_format is not None:
                txt = value_format.format(by)
            else:
                txt = f"{by:.3f}"
            plt.text(bx, by, txt, ha="center", va="bottom", fontsize=8)

    plt.xticks(x, x_labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend(fontsize=9)

    if ylim is not None:
        plt.ylim(*ylim)

    plt.figtext(
        0.99, 0.01, FOOTER,
        ha="right", va="bottom", fontsize=8
    )

    plt.tight_layout(rect=[0.04, 0.08, 0.97, 0.95])
    out_path = os.path.join(RESULT_DIR, filename)
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"Saved: {out_path}")


def main():
    print("Collecting CIFAR-10 metrics for ResNet-18 and MobileNetV2 ...")

    # ---- ResNet-18: eval_resnet18_c10_all ----
    r_metrics = collect_resnet_metrics()
    # r_metrics["names"] = ["Baseline FP32", "Pruned FP32", "Pruned + INT8"]
    r_names = r_metrics["names"]
    r_acc = r_metrics["accuracy"]
    r_size = r_metrics["size_mb"]
    r_lat = r_metrics["latency_s"]

    # baseline + pruned50
    # [Baseline, Pruned, Quant]
    r_acc_base = r_acc[0]
    r_acc_p50 = r_acc[1]
    r_size_base = r_size[0]
    r_size_p50 = r_size[1]
    r_lat_base = r_lat[0]
    r_lat_p50 = r_lat[1]

    # ---- MobileNetV2: eval_mobilenetv2_c10_all ----
    m_metrics = collect_metrics_mnet_c10()
    # m_metrics["names"] = ["MobileNetV2 FP32", "MobileNetV2 Pruned 50%"]
    m_acc = m_metrics["accuracy"]
    m_size = m_metrics["size_mb"]
    m_lat = m_metrics["latency_s"]

    m_acc_base = m_acc[0]
    m_acc_p50 = m_acc[1]
    m_size_base = m_size[0]
    m_size_p50 = m_size[1]
    m_lat_base = m_lat[0]
    m_lat_p50 = m_lat[1]

    model_labels = ["ResNet-18", "MobileNetV2"]

    # Accuracy
    acc_base = [r_acc_base * 100.0, m_acc_base * 100.0]
    acc_p50 = [r_acc_p50 * 100.0, m_acc_p50 * 100.0]

    # Size: MB
    size_base = [r_size_base, m_size_base]
    size_p50 = [r_size_p50, m_size_p50]

    # Latency: ms
    lat_base = [r_lat_base * 1000.0, m_lat_base * 1000.0]
    lat_p50 = [r_lat_p50 * 1000.0, m_lat_p50 * 1000.0]

    def compute_ylim(vs, margin_ratio=0.1):
        vmin, vmax = min(vs), max(vs)
        span = vmax - vmin
        if span == 0:
            span = max(1e-3, abs(vmax) * 0.1)
        return (vmin - margin_ratio * span, vmax + margin_ratio * span)

    # Accuracy
    acc_all = acc_base + acc_p50
    acc_ylim = compute_ylim(acc_all, margin_ratio=0.1)

    # Size
    size_all = size_base + size_p50
    size_ylim = compute_ylim(size_all, margin_ratio=0.1)

    # Latency
    lat_all = lat_base + lat_p50
    lat_ylim = compute_ylim(lat_all, margin_ratio=0.1)

    print("Generating cross-model comparison plots...")

    # 1) Accuracy
    grouped_bar(
        x_labels=model_labels,
        group_values=[acc_base, acc_p50],
        variant_labels=["Base", "Pruned 50%"],
        ylabel="Accuracy (%)",
        title="Accuracy vs Model & Pruning\nCIFAR-10 · ResNet-18 vs MobileNetV2",
        filename=f"{PREFIX}_accuracy.png",
        value_format="{:.1f}%",
        ylim=acc_ylim,
    )

    # 2) Model size
    grouped_bar(
        x_labels=model_labels,
        group_values=[size_base, size_p50],
        variant_labels=["Base", "Pruned 50%"],
        ylabel="Model size (MB)",
        title="Model Size vs Model & Pruning\nCIFAR-10 · ResNet-18 vs MobileNetV2",
        filename=f"{PREFIX}_modelsize.png",
        value_format="{:.3f} MB",
        ylim=size_ylim,
    )

    # 3) Latency
    grouped_bar(
        x_labels=model_labels,
        group_values=[lat_base, lat_p50],
        variant_labels=["Base", "Pruned 50%"],
        ylabel="Latency (ms / image, CPU)",
        title="Inference Latency vs Model & Pruning (CPU)\nCIFAR-10 · ResNet-18 vs MobileNetV2",
        filename=f"{PREFIX}_latency.png",
        value_format="{:.3f} ms",
        ylim=lat_ylim,
    )

    print("Done. Cross-model comparison plots saved in 'results/'.")


if __name__ == "__main__":
    main()
