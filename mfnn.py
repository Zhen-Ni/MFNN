#!/usr/bin/env python3

import typing
import torch

TensorFunc = typing.Callable[[torch.Tensor], torch.Tensor]

__all__ = ('FCNN',)


class BasicBlock(torch.nn.Module):
    """Basic block for a fully-connected layer."""

    def __init__(self, in_features: int,
                 out_features: int,
                 activation: type[torch.nn.Module] | None):
        super().__init__()
        self.linear_layer = torch.nn.Linear(in_features, out_features)
        if activation is None:
            activation = torch.nn.Identity
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
                 activation: type[torch.nn.Module] | None = torch.nn.ReLU
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

        self.nonlinear_nn = FCNN(in_features + out_features,
                                 out_features,
                                 midlayer_features,
                                 activation)
        self.linear_nn = FCNN(in_features + out_features,
                              out_features,
                              [],
                              torch.nn.Identity)
        self.nonlinear_ratio = torch.nn.Parameter(torch.Tensor(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_low = self.lfnn(x)
        x_comb = torch.concat([x, y_low], dim=1)
        y_nonlinear = self.nonlinear_nn(x_comb)
        y_linear = self.linear_nn(x_comb)
        y = (y_nonlinear * self.nonlinear_ratio +
             y_linear * (1 - self.nonlinear_ratio))
        return y


if __name__ == '__main__':

    x = torch.linspace(0, 1, 101).reshape(-1, 1)

    lfnn = FCNN(1, 1, [16, 16, 16], activation=torch.nn.Tanh)
    hfnn = HFNN(lfnn, 1, 1, [32, 32])

    ylow = lfnn(x)
    yhigh = hfnn(x)
