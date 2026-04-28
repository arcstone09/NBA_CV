import torch
import cv2
import numpy as np
from pathlib import Path
import pandas as pd
from torch.utils.data import DataLoader


BASE_DIR = Path("/workspace/NBA_CV")
input_path = BASE_DIR / "data" / "curry_24_crop_data"
csv_path = BASE_DIR / "data" / "curry_24_split.csv"


def resize_with_padding(frame, size=112):
    h, w, _ = frame.shape
    scale = size / max(h, w)

    new_h = int(h * scale)
    new_w = int(w * scale)

    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((size, size, 3), dtype=np.uint8)

    y_offset = (size - new_h) // 2
    x_offset = (size - new_w) // 2

    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    return canvas


def load_video_frames(video_path, num_frames=16, size=112):
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = resize_with_padding(frame, size=size)
        frames.append(frame)

    cap.release()

    if len(frames) == 0:
        raise RuntimeError(f"No frames found: {video_path}")

    total_frames = len(frames)
    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)
    sampled = [frames[i] for i in indices]

    frames_np = np.stack(sampled)
    frames_np = frames_np.astype(np.float32) / 255.0

    # (T, H, W, C) -> (C, T, H, W)
    frames_tensor = torch.from_numpy(frames_np).permute(3, 0, 1, 2)

    return frames_tensor


class ShotDataset(torch.utils.data.Dataset):
    def __init__(self, df, base_path, num_frames=16, size=112):
        self.df = df.reset_index(drop=True)
        self.base_path = Path(base_path)
        self.num_frames = num_frames
        self.size = size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = self.base_path / row["VIDEO_PATH"]

        frames = load_video_frames(
            video_path,
            num_frames=self.num_frames,
            size=self.size,
        )

        label = torch.tensor(row["LABEL"], dtype=torch.float32)

        return frames, label


def get_dataloaders(batch_size=8, num_frames=16, size=112):
    df = pd.read_csv(csv_path)

    train_df = df[df["SPLIT"] == "train"]
    val_df = df[df["SPLIT"] == "val"]
    test_df = df[df["SPLIT"] == "test"]

    train_dataset = ShotDataset(train_df, input_path, num_frames, size)
    val_dataset = ShotDataset(val_df, input_path, num_frames, size)
    test_dataset = ShotDataset(test_df, input_path, num_frames, size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = get_dataloaders()

    frames, labels = next(iter(train_loader))

    print("frames shape:", frames.shape)
    print("labels shape:", labels.shape)
    print("labels dtype:", labels.dtype)
    print(labels)