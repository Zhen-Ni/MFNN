#!/usr/bin/env python3

from __future__ import annotations
import typing
import torch

TensorFunc = typing.Callable[[torch.Tensor], torch.Tensor]

__all__ = ('XYDataSet', 'FCNN', 'HFNN', 'MFNN')


class XYDataSet(torch.utils.data.Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        if len(x) != len(y):
            raise ValueError('size of x and y not match')
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx) -> torch.Tensor:
        return self.x[idx], self.y[idx]


class BasicBlock(torch.nn.Module):
    """Basic block for a fully-connected layer."""

    def __init__(self, in_features: int,
                 out_features: int,
                 activation: type[torch.nn.Module]):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features, out_features)
        self.activation = activation()

    def forward(self, x):
        x = self.linear_layer(x)
        x = self.activation(x)
        return x


class FCNN(torch.nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 midlayer_features: list[int],
                 activation: type[torch.nn.Module] = torch.nn.ReLU
                 ):
        super().__init__()

        layer_sizes = [in_features] + midlayer_features

        self.layers = torch.nn.Sequential(*[
            BasicBlock(layer_sizes[i], layer_sizes[i+1], activation)
            for i in range(len(midlayer_features))
        ])
        self.fc = torch.nn.Linear(layer_sizes[-1], out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        x = self.fc(x)
        return x


class HFNN(torch.nn.Module):
    def __init__(self,
                 lfnn: torch.nn.Module | TensorFunc,
                 in_features: int,
                 out_features: int,
                 midlayer_features: list[int],
                 activation: type[torch.nn.Module] | None = None):
        activation = torch.nn.Tanh if activation is None else activation
        super().__init__()
        # Avoid updating the trained low-fidelity model during training.
        object.__setattr__(self, 'lfnn', lfnn)
        self.lfnn: torch.nn.Module | TensorFunc

        self.layers = FCNN(in_features + out_features,
                           out_features,
                           midlayer_features,
                           activation)
        self.shortcut = FCNN(in_features + out_features,
                             out_features,
                             [],
                             torch.nn.Identity)

        # Use He Kaiming's normal initialization
        for m in self.modules():
            for name, parameter in m.named_parameters():
                if name == 'weight':
                    torch.nn.init.kaiming_normal_(parameter)
                elif name == 'bias':
                    torch.nn.init.zeros_(parameter)
                else:
                    assert "impossible!"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_low = self.lfnn(x)
        x_comb = torch.concat([x, y_low], dim=1)
        y_layer = self.layers(x_comb)
        y_shortcut = self.shortcut(x_comb)
        y = y_layer + y_shortcut
        return y


class MFNN():
    """A flexible network for regression problem of multi-fidelity data.

    This class contains two networks, for fitting low-fidelity data and
    high fidelity data respectively.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 backbone_layers: list[int],
                 layers_low: list[int],
                 layers_high: list[int],
                 activation: type[torch.nn.Module] | None = None):
        activation = torch.nn.Tanh if activation is None else activation

        layer_sizes = [in_features] + backbone_layers
        self.backbone = torch.nn.Sequential(*[
            BasicBlock(layer_sizes[i], layer_sizes[i+1], activation)
            for i in range(len(backbone_layers))
        ])

        self.fc_low = FCNN(layer_sizes[-1],
                           out_features,
                           layers_low,
                           activation)
        self.fc_high = FCNN(out_features + layer_sizes[-1],
                            out_features,
                            layers_high,
                            activation)
        self.shortcut = FCNN(out_features + layer_sizes[-1],
                             out_features,
                             [],
                             torch.nn.Identity)

        # Use He Kaiming's normal initialization
        for m in [self.backbone, self.fc_low, self.fc_high, self.shortcut]:
            for name, parameter in m.named_parameters():
                if name == 'weight':
                    torch.nn.init.kaiming_normal_(parameter)
                elif name == 'bias':
                    torch.nn.init.zeros_(parameter)
                else:
                    assert "impossible!"

        self.low = MFNNLow(self.backbone, self.fc_low)
        self.high = MFNNHigh(self.low, self.fc_high, self.shortcut)


class MFNNLow(torch.nn.Module):
    def __init__(self,
                 backbone: torch.nn.Module,
                 fc_low: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        self.fc_low = fc_low

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        y = self.fc_low(features)
        return y


class MFNNHigh(torch.nn.Module):
    def __init__(self,
                 mfnnlow: torch.nn.Module,
                 fc_high: torch.nn.Module,
                 shortcut: torch.nn.Module):
        super().__init__()
        # Avoid updating the trained low-fidelity model during training.
        object.__setattr__(self, 'mfnnlow', mfnnlow)
        self.mfnnlow: torch.nn.Module
        self.fc_high = fc_high
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.mfnnlow.backbone(x)
        y_low = self.mfnnlow.fc_low(features)
        x_combine = torch.concat([y_low.reshape(-1, 1), features], dim=1)
        y_fc = self.fc_high(x_combine)
        y_shortcut = self.shortcut(x_combine)
        y = y_fc + y_shortcut
        return y


if __name__ == '__main__':

    x = torch.linspace(0, 1, 101).reshape(-1, 1)

    lfnn = FCNN(1, 1, [16, 16, 16], activation=torch.nn.Tanh)
    hfnn = HFNN(lfnn, 1, 1, [32, 32])
    mfnn = MFNN(1, 1, [16], [16, 16], [16], activation=torch.nn.Tanh)

    ylow = lfnn(x)
    yhigh = hfnn(x)
    ylow = mfnn.low()(x)
    yhigh = mfnn.high()(x)
