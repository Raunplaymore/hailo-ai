"""Event detection model built on MobileNetV2 + LSTM."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.autograd import Variable

from .backbone.mobilenet_v2 import MobileNetV2


class EventDetector(nn.Module):
    """CNN + LSTM event detector for GolfDB swings."""

    def __init__(
        self,
        pretrain: bool,
        width_mult: float,
        lstm_layers: int,
        lstm_hidden: int,
        bidirectional: bool = True,
        dropout: bool = True,
        weights_path: Optional[str | Path] = None,
    ) -> None:
        super().__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout

        net = MobileNetV2(width_mult=width_mult)
        if pretrain:
            weights_file = Path(weights_path) if weights_path else Path("mobilenet_v2.pth.tar")
            state_dict_mobilenet = torch.load(weights_file, map_location="cpu")
            net.load_state_dict(state_dict_mobilenet)

        self.cnn = nn.Sequential(*list(net.children())[0][:19])
        feature_dim = int(1280 * width_mult if width_mult > 1.0 else 1280)
        self.rnn = nn.LSTM(
            feature_dim,
            self.lstm_hidden,
            self.lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.lin = nn.Linear(2 * self.lstm_hidden if self.bidirectional else self.lstm_hidden, 9)
        self.drop = nn.Dropout(0.5) if self.dropout else None

    def init_hidden(self, batch_size: int, device: torch.device) -> Tuple[Variable, Variable]:
        """Initialize LSTM hidden states on the given device."""
        num_dirs = 2 if self.bidirectional else 1
        h0 = torch.zeros(num_dirs * self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        c0 = torch.zeros(num_dirs * self.lstm_layers, batch_size, self.lstm_hidden, device=device)
        return Variable(h0, requires_grad=True), Variable(c0, requires_grad=True)

    def forward(self, x: torch.Tensor, device: torch.device) -> torch.Tensor:
        batch_size, timesteps, C, H, W = x.size()
        hidden = self.init_hidden(batch_size, device=device)

        # CNN forward
        c_in = x.view(batch_size * timesteps, C, H, W)
        c_out = self.cnn(c_in)
        c_out = c_out.mean(3).mean(2)
        if self.drop:
            c_out = self.drop(c_out)

        # LSTM forward
        r_in = c_out.view(batch_size, timesteps, -1)
        r_out, _ = self.rnn(r_in, hidden)
        out = self.lin(r_out)
        return out.view(batch_size * timesteps, 9)


__all__ = ["EventDetector"]
