import torch
import torch.nn as nn
from torchvision.models import resnet18

# Reuse dataloaders + evaluation from the baseline training script
from train_resnet18_c10_baseline import get_loaders, evaluate


def load_fp32_model(path, num_classes=10):
    """
    Load a standard FP32 ResNet-18 model trained on CIFAR-10.
    This function restores the model architecture and loads its parameters.
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def main():
    """
    Apply dynamic quantization to the pruned FP32 model and evaluate accuracy.
    Dynamic quantization only quantizes Linear layers and is performed on CPU.
    """

    device = "cpu"
    _, testloader = get_loaders()

    # Load the pruned FP32 model (50% pruning on CIFAR-10)
    fp32_path = "models/resnet18_c10_pruned50.pth"
    fp32_model = load_fp32_model(fp32_path)
    fp32_model.eval()

    # Apply dynamic INT8 quantization to Linear layers
    int8_model = torch.quantization.quantize_dynamic(
        fp32_model, {nn.Linear}, dtype=torch.qint8
    )
    int8_model.eval()

    # Evaluate the models
    acc_fp32 = evaluate(fp32_model.to(device), testloader, device)
    acc_int8 = evaluate(int8_model.to(device), testloader, device)

    print(f"Pruned FP32 accuracy:   {acc_fp32:.4f}")
    print(f"Pruned + INT8 accuracy: {acc_int8:.4f}")

    # Save the entire quantized model (not state_dict)
    save_path = "models/resnet18_c10_pruned50_int8.pth"
    torch.save(int8_model, save_path)
    print(f"Quantized model saved as {save_path}")


if __name__ == "__main__":
    main()
