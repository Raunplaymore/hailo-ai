"""Inference script for GolfDB event detector on a single video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms

from src.dataset.golfdb_dataset import Normalize, ToTensor
from src.models.event_detector import EventDetector
from src.utils.paths import get_data_root


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> EventDetector:
    model = EventDetector(
        pretrain=False,
        width_mult=1.0,
        lstm_layers=1,
        lstm_hidden=256,
        bidirectional=True,
        dropout=False,
        weights_path=None,
    )
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def load_video_frames(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    if not frames:
        raise FileNotFoundError(f"No frames read from video: {video_path}")
    return np.asarray(frames)  # (num_frames, H, W, C)


def predict_frames(model: EventDetector, frames: np.ndarray, device: torch.device) -> None:
    # Apply same transforms as dataset
    transform = transforms.Compose(
        [
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dummy_labels = np.zeros(len(frames))
    sample = {"images": frames, "labels": dummy_labels}
    sample = transform(sample)
    images = sample["images"]  # (T, C, H, W)

    images = images.unsqueeze(0).to(device)  # (1, T, C, H, W)
    with torch.no_grad():
        logits = model(images, device=device)  # (1*T, num_classes)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        confidences = probs.max(dim=1).values

    preds = preds.view(-1).cpu().numpy()
    confidences = confidences.view(-1).cpu().numpy()
    for idx, (p, conf) in enumerate(zip(preds, confidences)):
        print(f"Frame {idx:04d}: class={p}, confidence={conf:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference on a video with GolfDB event detector")
    parser.add_argument(
        "--video",
        type=str,
        default=str(get_data_root() / "test_video.mp4"),
        help="Path to video file (default: data/test_video.mp4)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (.pth or .pth.tar)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Using device: {device}")

    video_path = Path(args.video)
    model = load_model(Path(args.checkpoint), device)
    frames = load_video_frames(video_path)
    predict_frames(model, frames, device)


if __name__ == "__main__":
    main()
