import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import mobilenet_v2
import os
from tqdm import tqdm

from train_resnet18_c10_baseline import get_loaders, evaluate


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training MobileNetV2 on CIFAR-10, device: {device}")

    trainloader, testloader = get_loaders()

    model = mobilenet_v2(weights=None)
    # Replace classifier head for 10 classes (CIFAR-10)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.1,
        momentum=0.9,
        weight_decay=5e-4,
    )

    epochs = 30
    os.makedirs("models", exist_ok=True)
    best_acc = 0.0
    save_path = "models/mobilenetv2_c10_base.pth"

    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(trainloader, desc=f"[MobileNetV2] Epoch {epoch+1}/{epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        acc = evaluate(model, testloader, device)
        print(f"[CIFAR-10][MobileNetV2][Epoch {epoch+1}] Test Acc = {acc:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), save_path)
            print(f"✓ Saved checkpoint to {save_path}")

    print("[CIFAR-10][MobileNetV2] Best accuracy:", best_acc)


if __name__ == "__main__":
    main()
