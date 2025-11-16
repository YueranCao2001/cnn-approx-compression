import os
import time
import torch
import torch.nn as nn
from torchvision.models import resnet18

from train_baseline import get_loaders, evaluate

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
    """
    model.to(device)
    model.eval()
    x = torch.randn(1, 3, 32, 32, device=device)

    # Warm-up
    with torch.no_grad():
        for _ in range(10):
            model(x)

        start = time.time()
        for _ in range(runs):
            model(x)
        end = time.time()

    return (end - start) / runs  # average seconds per image

def main():
    device = "cpu"

    # File size comparison
    print("Baseline FP32 size:        {:.4f} MB".format(get_file_size("models/resnet18_base.pth")))
    print("Pruned FP32 size:          {:.4f} MB".format(get_file_size("models/resnet18_pruned.pth")))
    print("Pruned + INT8 size:        {:.4f} MB".format(get_file_size("models/resnet18_pruned_int8.pth")))

    _, testloader = get_loaders()

    # Load FP32 models
    base = load_fp32_model("models/resnet18_base.pth")
    pruned = load_fp32_model("models/resnet18_pruned.pth")

    # Load the INT8 model (saved as a full model, not state_dict)
    quantized = torch.load("models/resnet18_pruned_int8.pth", map_location=device)
    quantized.eval()

    # Accuracy evaluation
    acc_base = evaluate(base.to(device), testloader, device)
    acc_pruned = evaluate(pruned.to(device), testloader, device)
    acc_quant = evaluate(quantized.to(device), testloader, device)

    print("\nAccuracy on CIFAR-10 (CPU):")
    print(f"  Baseline FP32:          {acc_base:.4f}")
    print(f"  Pruned FP32:            {acc_pruned:.4f}")
    print(f"  Pruned + INT8:          {acc_quant:.4f}")

    # Latency test
    t_base = speed_test(base, device)
    t_pruned = speed_test(pruned, device)
    t_quant = speed_test(quantized, device)

    print("\nAverage inference time per image (CPU):")
    print(f"  Baseline FP32:          {t_base*1000:.3f} ms")
    print(f"  Pruned FP32:            {t_pruned*1000:.3f} ms")
    print(f"  Pruned + INT8:          {t_quant*1000:.3f} ms")

if __name__ == "__main__":
    main()
