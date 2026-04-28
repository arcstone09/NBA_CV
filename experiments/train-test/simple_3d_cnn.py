import torch
import torch.nn as nn

from data_load import get_dataloaders


class Simple3DCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv3d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),

            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),

            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

        self.classifier = nn.Linear(64, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        logit = self.classifier(x)
        return logit.squeeze(1)


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders(
        batch_size=8,
        num_frames=16,
        size=112,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = Simple3DCNN().to(device)

    frames, labels = next(iter(train_loader))

    frames = frames.to(device)
    labels = labels.to(device)

    logits = model(frames)

    print("frames:", frames.shape)
    print("labels:", labels.shape)
    print("logits:", logits.shape)
    print("logits:", logits)