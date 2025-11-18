import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torchvision.models import resnet18
import os
from tqdm import tqdm


def get_loaders_c100(batch_size=128):
    """Return CIFAR-100 train/test dataloaders."""
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
    ])

    transform_test = T.Compose([
        T.ToTensor(),
    ])

    trainset = torchvision.datasets.CIFAR100(
        root="./data", train=True, download=True, transform=transform_train
    )
    testset = torchvision.datasets.CIFAR100(
        root="./data", train=False, download=True, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=256, shuffle=False
    )
    return trainloader, testloader


def evaluate(model, loader, device):
    """Evaluate accuracy on given loader."""
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(1)
            total += y.size(0)
            correct += (pred == y).sum().item()

    return correct / total


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training ResNet-18 on CIFAR-100, device: {device}")

    trainloader, testloader = get_loaders_c100()

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 100)  # 100 classes
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4
    )

    epochs = 30
    os.makedirs("models", exist_ok=True)

    best_acc = 0
    save_path = "models/resnet18_c100_base.pth"

    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(trainloader, desc=f"[C100] Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, testloader, device)
        print(f"[CIFAR-100][Epoch {epoch+1}] Test Acc = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved checkpoint to {save_path}")

    print("[CIFAR-100] Best accuracy:", best_acc)


if __name__ == "__main__":
    main()
