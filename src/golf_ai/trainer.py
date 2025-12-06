"""Trainer utilities for GolfDB event detection.

Intended to be shared by local/Colab runners. Use the root `train.py` as the CLI entrypoint.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple
import re

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset.golfdb_dataset import GolfDB, Normalize, ToTensor
from src.models.event_detector import EventDetector
from src.utils.paths import get_data_path
from src.utils.video_utils import AverageMeter, freeze_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train GolfDB event detector")
    parser.add_argument("--split", type=int, default=1, help="Train split index")
    parser.add_argument("--iterations", type=int, default=2000, help="Training iterations")
    parser.add_argument("--it-save", type=int, default=100, help="Checkpoint save interval")
    parser.add_argument("--num-workers", type=int, default=6, help="DataLoader workers")
    parser.add_argument("--seq-length", type=int, default=64, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=22, help="Batch size")
    parser.add_argument("--freeze-layers", type=int, default=10, help="Backbone layers to freeze")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--width-mult", type=float, default=1.0, help="MobileNetV2 width multiplier")
    parser.add_argument("--lstm-layers", type=int, default=1, help="LSTM layers")
    parser.add_argument("--lstm-hidden", type=int, default=256, help="LSTM hidden size")
    parser.add_argument("--no-pretrain", action="store_true", help="Disable MobileNetV2 pretrain weights")
    parser.add_argument("--weights-path", type=str, default=None, help="Path to mobilenet_v2 weights")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Use small subset, fewer iterations/steps, and verbose logging for local debugging",
    )
    parser.add_argument(  # ← 여기 추가
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint (.pth or .pth.tar) to resume training from",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloader(
    split: int,
    seq_length: int,
    batch_size: int,
    num_workers: int,
    train: bool,
    subset_size: int | None = None,
) -> Tuple[DataLoader, torch.Tensor]:
    split_root = get_data_path("golf_db")
    data_file = split_root / f"train_split_{split}.pkl" if train else f"test_split_{split}.pkl"
    vid_dir = split_root / "videos_160"
    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet mean/std
    dataset = GolfDB(
        data_file=data_file,
        vid_dir=vid_dir,
        seq_length=seq_length,
        transform=transforms.Compose([ToTensor(), norm]),
        train=train,
        subset_size=subset_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        drop_last=train,
    )
    class_weights = torch.FloatTensor([1 / 8] * 8 + [1 / 35])
    return loader, class_weights


def apply_debug_overrides(args: argparse.Namespace) -> argparse.Namespace:
    """Mutate args with lightweight settings for local debugging."""
    args.iterations = min(args.iterations, 50)
    args.it_save = min(args.it_save, 25)
    args.num_workers = 0
    args.seq_length = min(args.seq_length, 16)
    args.batch_size = min(args.batch_size, 4)
    args.freeze_layers = min(args.freeze_layers, 5)
    args.debug = True
    return args


class Trainer:
    """Trainer wrapper for GolfDB event detection."""

    def __init__(self, args: argparse.Namespace) -> None:
        if args.debug:
            args = apply_debug_overrides(args)
            print("[debug] Running with lightweight settings:", args)
        self.args = args
        self.device = get_device()
        print(f"Using device: {self.device}")
        self.split = args.split
        self.iterations = args.iterations

        self.loader, class_weights = build_dataloader(
            split=args.split,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train=True,
            subset_size=128 if args.debug else None,
        )
        self.class_weights = class_weights.to(self.device)

        self.model = EventDetector(
            pretrain=not args.no_pretrain,
            width_mult=args.width_mult,
            lstm_layers=args.lstm_layers,
            lstm_hidden=args.lstm_hidden,
            bidirectional=True,
            dropout=False,
            weights_path=args.weights_path,
        )
        freeze_layers(args.freeze_layers, self.model)
        self.model.train()
        self.model.to(self.device)

        self.criterion = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr
        )
        self.losses = AverageMeter()

        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

        self.start_iter = 0
        if args.resume is not None:
            ckpt_path = Path(args.resume)
            if not ckpt_path.exists():
                raise FileNotFoundError(f"Resume checkpoint not found: {ckpt_path}")
            print(f"[INFO] Loading checkpoint from: {ckpt_path}")

            ckpt = torch.load(ckpt_path, map_location=self.device)

            # swingnet_XXXX.pth.tar 형식 (dict) 지원
            if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
                if "optimizer_state_dict" in ckpt:
                    try:
                        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                        print("[INFO] Optimizer state loaded from checkpoint.")
                    except Exception as e:
                        print(f"[WARN] Failed to load optimizer state: {e}")
                self.start_iter = int(ckpt.get("iter", 0))
                print(f"[INFO] Resuming from iteration {self.start_iter}")
            else:
                # 단순 state_dict(.pth) 형식도 지원
                self.model.load_state_dict(ckpt)
                # 파일명에서 iter 숫자 추출 (예: split1_iter1200.pth)
                m = re.search(r"iter(\d+)", ckpt_path.stem)
                if m:
                    self.start_iter = int(m.group(1))
                    print(f"[INFO] Resuming from iteration {self.start_iter} (parsed from filename)")
                else:
                    print("[WARN] Could not determine starting iteration. Starting from 0.")

    def run(self) -> None:
        i = self.start_iter
        args = self.args
        while i < args.iterations:
            for sample in self.loader:
                images, labels = sample["images"].to(self.device), sample["labels"].to(self.device)
                logits = self.model(images, device=self.device)
                labels = labels.view(args.batch_size * args.seq_length)
                loss = self.criterion(logits, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.losses.update(loss.item(), images.size(0))
                self.optimizer.step()
                # ----------------------------
                # CHECKPOINT SAVE BLOCK
                # ----------------------------

                # iteration은 0부터 시작하므로 +1
                iter_num = i + 1

                # 매 100 iteration마다 저장
                if iter_num % 100 == 0:
                    ckpt_dir = Path("checkpoints")
                    ckpt_dir.mkdir(exist_ok=True)

                    ckpt_path = ckpt_dir / f"split{self.split}_iter{iter_num}.pth"
                    torch.save(self.model.state_dict(), ckpt_path)
                    print(f"[INFO] Saved checkpoint → {ckpt_path}")

                # 마지막 iteration에서 최종본 저장
                if iter_num == self.iterations:
                    ckpt_dir = Path("checkpoints")
                    ckpt_dir.mkdir(exist_ok=True)

                    final_path = ckpt_dir / f"split{self.split}_final.pth"
                    torch.save(self.model.state_dict(), final_path)
                    print(f"[INFO] Saved FINAL checkpoint → {final_path}")
                # ----------------------------
                if i % max(1, args.it_save // 4) == 0 or args.debug:
                    print(f"Iteration: {i}\tLoss: {self.losses.val:.4f} (avg {self.losses.avg:.4f})")
                i += 1
                if i % args.it_save == 0:
                    ckpt = {
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "model_state_dict": self.model.state_dict(),
                        "iter": i,
                    }
                    torch.save(ckpt, self.models_dir / f"swingnet_{i}.pth.tar")
                if i >= args.iterations:
                    break


__all__ = ["Trainer", "parse_args"]
