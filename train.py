"""Training entrypoint for GolfDB event detection (CLI)."""

from __future__ import annotations

from src.golf_ai.trainer import Trainer, parse_args


def main() -> None:
    args = parse_args()
    trainer = Trainer(args)
    trainer.run()


if __name__ == "__main__":
    main()
