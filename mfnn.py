#!/usr/bin/env python3

import typing
import torch

TensorFunc = typing.Callable[[torch.Tensor], torch.Tensor]

__all__ = ('XYDataSet', 'FCNN', 'HFNN')


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

        # Use He Kaiming's normal initialization
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                for name, parameter in m.named_parameters():
                    if name == 'weight':
                        torch.nn.init.kaiming_normal_(parameter)
                    elif name == 'bias':
                        torch.nn.init.zeros_(parameter)
                    else:
                        assert "impossible!"

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_low = self.lfnn(x)
        x_comb = torch.concat([x, y_low], dim=1)
        y_layer = self.layers(x_comb)
        y_shortcut = self.shortcut(x_comb)
        y = y_layer + y_shortcut
        return y


if __name__ == '__main__':

    x = torch.linspace(0, 1, 101).reshape(-1, 1)

    lfnn = FCNN(1, 1, [16, 16, 16], activation=torch.nn.Tanh)
    hfnn = HFNN(lfnn, 1, 1, [32, 32])

    ylow = lfnn(x)
    yhigh = hfnn(x)
