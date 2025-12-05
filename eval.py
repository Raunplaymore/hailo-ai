"""Evaluation script for GolfDB event detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset.golfdb_dataset import GolfDB, Normalize, ToTensor
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


def build_val_loader(split: int, device: torch.device) -> tuple[DataLoader, torch.Tensor]:
    data_root = get_data_root() / "golf_db"
    val_file = data_root / f"val_split_{split}.pkl"
    vid_dir = data_root / "videos_160"
    transform = transforms.Compose(
        [
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    dataset = GolfDB(
        data_file=val_file,
        vid_dir=vid_dir,
        seq_length=64,
        transform=transform,
        train=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    class_weights = torch.FloatTensor([1 / 8] * 8 + [1 / 35]).to(device)
    return loader, class_weights


def evaluate(model: EventDetector, loader: DataLoader, device: torch.device, class_weights: torch.Tensor) -> None:
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    total_loss = 0.0
    total_frames = 0
    correct = 0

    with torch.no_grad():
        for sample in loader:
            images, labels = sample["images"].to(device), sample["labels"].to(device)
            logits = model(images, device=device)  # (batch*timesteps, num_classes)

            labels = labels.view(-1)
            logits = logits.view(-1, logits.size(-1))

            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.numel()
            total_frames += labels.numel()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()

    avg_loss = total_loss / total_frames if total_frames else 0.0
    acc = correct / total_frames if total_frames else 0.0
    print(f"Validation accuracy: {acc:.4f}")
    print(f"Average loss: {avg_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GolfDB event detector")
    parser.add_argument("--split", type=int, default=1, help="Validation split index (e.g., 1)")
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

    loader, class_weights = build_val_loader(args.split, device)
    model = load_model(Path(args.checkpoint), device)
    evaluate(model, loader, device, class_weights)


if __name__ == "__main__":
    main()
