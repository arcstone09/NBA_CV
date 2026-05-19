import torch
import torch.nn as nn

from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights

from data_load import get_dataloaders


class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        weights = R2Plus1D_18_Weights.DEFAULT
        self.model = r2plus1d_18(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, 1)

    def forward(self, x):
        logit = self.model(x)
        return logit.squeeze(1)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        num_frames=16,
        size=112,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    if torch.cuda.is_available():
        print("gpu:", torch.cuda.get_device_name(0))

    model = Simple3DCNN().to(device)

    frames, labels = next(iter(train_loader))

    frames = frames.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    logits = model(frames)

    print("frames:", frames.shape)
    print("labels:", labels.shape)
    print("logits:", logits.shape)
    print("logits:", logits)