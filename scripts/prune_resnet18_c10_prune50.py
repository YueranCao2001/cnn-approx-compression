import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models import resnet18
import os

# Import dataloader + evaluation from baseline script
from train_resnet18_c10_baseline import get_loaders, evaluate


def load_model(path="models/resnet18_c10_base.pth", num_classes=10):
    """
    Load the baseline FP32 ResNet-18 checkpoint trained on CIFAR-10.
    This model will be used as the starting point for pruning.
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model


def apply_global_pruning(model, amount=0.5):
    """
    Apply global unstructured pruning to ALL Conv2d + Linear layers.
    amount: fraction of weights to prune (0.5 = prune 50%).
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            parameters_to_prune.append((module, "weight"))

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    return model


def remove_pruning_reparam(model):
    """
    Remove pruning reparameterization (`weight_orig`, `mask`) 
    and make the remaining weights permanent.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, "weight_orig"):
                prune.remove(module, "weight")
    return model


def main(prune_amount=0.5, epochs=5):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CIFAR-10 dataloaders
    trainloader, testloader = get_loaders()

    # Load baseline model
    model = load_model().to(device)

    # Apply pruning
    print(f"\nApplying global pruning (amount={prune_amount})...")
    model = apply_global_pruning(model, amount=prune_amount)

    # Fine-tune the model after pruning
    print(f"Fine-tuning for {epochs} epochs...")
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
        print(f"[Fine-tune {epoch+1}/{epochs}] Test Accuracy: {acc:.4f}")

    # Remove pruning masks
    model = remove_pruning_reparam(model)

    # Save pruned checkpoint
    os.makedirs("models", exist_ok=True)
    save_path = f"models/resnet18_c10_pruned50.pth"
    torch.save(model.state_dict(), save_path)

    print(f"\n✓ Pruned model saved to {save_path}")


if __name__ == "__main__":
    main()
