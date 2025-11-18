import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models import resnet18
import os

from train_resnet18_c100_baseline import get_loaders_c100, evaluate


def load_model(path="models/resnet18_c100_base.pth", num_classes=100):
    """Load CIFAR-100 baseline FP32 ResNet-18 checkpoint."""
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def apply_global_pruning(model, amount=0.5):
    """
    Apply global unstructured pruning on all Conv2d + Linear layers.
    amount: fraction of weights to prune (0.5 = prune 50%).
    """
    parameters_to_prune = []
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    return model


def remove_pruning_reparam(model):
    """Remove pruning reparameterization and make remaining weights permanent."""
    for _, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, "weight_orig"):
                prune.remove(module, "weight")
    return model


def main(prune_amount=0.5, epochs=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[CIFAR-100] Using device: {device}")

    trainloader, testloader = get_loaders_c100()

    model = load_model().to(device)

    print(f"\n[CIFAR-100] Applying global pruning (amount={prune_amount})...")
    model = apply_global_pruning(model, amount=prune_amount)

    print(f"[CIFAR-100] Fine-tuning pruned model for {epochs} epochs...")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, testloader, device)
        print(f"[CIFAR-100][Fine-tune {epoch+1}/{epochs}] Test Accuracy: {acc:.4f}")

    model = remove_pruning_reparam(model)

    os.makedirs("models", exist_ok=True)
    save_path = "models/resnet18_c100_pruned50.pth"
    torch.save(model.state_dict(), save_path)
    print(f"\n✓ 50% pruned CIFAR-100 model saved to {save_path}")


if __name__ == "__main__":
    main()
