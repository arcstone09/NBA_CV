import torch
import torch.nn as nn
from torch.optim import Adam

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score
from sklearn.calibration import calibration_curve

from data_load import get_dataloaders
from simple_3d_cnn import Simple3DCNN


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []

    for frames, labels in train_loader:
        frames = frames.to(device)
        labels = labels.to(device).float()

        logits = model(frames)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()

        correct += (preds == labels).sum().item()
        total += labels.size(0)

        all_probs.extend(probs.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = None

    return avg_loss, acc, auc


def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            labels = labels.to(device).float()

            logits = model(frames)
            loss = criterion(logits, labels)

            total_loss += loss.item() * frames.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    acc = correct / total

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = None

    return avg_loss, acc, auc, np.array(all_probs), np.array(all_labels)


def plot_calibration_curve(labels, probs, save_path="calibration_curve.png"):
    prob_true, prob_pred = calibration_curve(
        labels,
        probs,
        n_bins=10,
        strategy="uniform"
    )

    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o", label="model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="perfect calibration")
    plt.xlabel("Predicted probability")
    plt.ylabel("Actual success rate")
    plt.title("Calibration Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        num_frames=16,
        size=112,
    )

    model = Simple3DCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-4)

    num_epochs = 20
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        train_loss, train_acc, train_auc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc, val_auc, val_probs, val_labels = evaluate(
            model, val_loader, criterion, device
        )

        print(
            f"Epoch [{epoch}/{num_epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"train_auc={train_auc if train_auc is not None else 'NA'} "
            f"val_loss={val_loss:.4f} "
            f"val_acc={val_acc:.4f} "
            f"val_auc={val_auc if val_auc is not None else 'NA'}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_simple_3d_cnn.pth")
            print("saved best model")

    model.load_state_dict(
        torch.load("best_simple_3d_cnn.pth", map_location=device)
    )

    test_loss, test_acc, test_auc, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device
    )

    print("Final Test")
    print(f"test_loss={test_loss:.4f}")
    print(f"test_acc={test_acc:.4f}")
    print(f"test_auc={test_auc if test_auc is not None else 'NA'}")

    plot_calibration_curve(
        test_labels,
        test_probs,
        save_path="test_calibration_curve.png"
    )

    print("saved calibration curve: test_calibration_curve.png")


if __name__ == "__main__":
    main()