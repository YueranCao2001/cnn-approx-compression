import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from torchvision.models import resnet18
from train_baseline import get_loaders, evaluate

def load_model(path="models/resnet18_base.pth", num_classes=10):
    """
    Load the baseline FP32 ResNet-18 model from disk.
    This model is the starting point for pruning.
    """
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(path, map_location="cpu")
    model.load_state_dict(state_dict)
    return model

def apply_global_pruning(model, amount=0.5):
    """
    Apply global unstructured pruning across all Conv2d and Linear layers.
    The 'amount' parameter determines the percentage of weights to prune.
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
    Remove pruning reparameterization and make masked weights permanent.
    After this, 'weight_orig' and pruning buffers are removed.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            if hasattr(module, "weight_orig"):
                prune.remove(module, "weight")
    return model

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    trainloader, testloader = get_loaders()

    # Load the baseline model
    model = load_model().to(device)

    # Apply pruning
    model = apply_global_pruning(model, amount=0.5)
    print("Global pruning applied (50% of weights pruned).")

    # Fine-tune the model to recover accuracy
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(5):
        model.train()
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, testloader, device)
        print(f"Fine-tuning epoch {epoch+1}/5 - Test accuracy: {acc:.4f}")

    # Remove pruning masks and save the final pruned model
    model = remove_pruning_reparam(model)
    torch.save(model.state_dict(), "models/resnet18_pruned.pth")
    print("Pruned model saved to models/resnet18_pruned.pth")

if __name__ == "__main__":
    main()
