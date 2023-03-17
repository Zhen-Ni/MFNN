#!/usr/bin/env python3

import logging
import torch

from mfnn import FCNN, HFNN, MFNN, XYDataSet
from trainer import Trainer

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=10)
plt.rc('mathtext', fontset='stix')


def func_low(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(8 * torch.pi * x)


def func_high(x: torch.Tensor) -> torch.Tensor:
    return (x - 2**.5) * func_low(x) ** 2


def figure1(x_low, y_low, x_high,  y_high, x, y):
    "Plot the multi-fidelity data along with true data."
    y_high = func_high(x_high)

    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.plot(x_low, y_low, 'o', color='None', markeredgecolor='b', label='low')
    ax.plot(x_high, y_high, 'rx', label='high')
    ax.plot(x, y, 'k:', label='true')
    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(x[0], x[-1])
    ax.grid()
    fig.tight_layout(pad=0)
    return fig


def figure2(x_low, y_low, x_pred, y_pred, x, y):
    "Plot the regression result by low-fidelity data."
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.plot(x_low, y_low, 'o', color='None', markeredgecolor='b', label='low')
    ax.plot(x_pred, y_pred, 'r', label='$y_{\mathrm{low}}$')
    ax.plot(x, y, 'k:', label='true')
    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(x[0], x[-1])
    ax.grid()
    fig.tight_layout(pad=0)
    return fig

def figure3(x_low, y_low, x_high, y_high, x_pred, y_pred, x, y):
    "Plot the regression result by low-fidelity data."
    fig = plt.figure(figsize=(3, 2.25))
    ax = fig.add_subplot(111)
    ax.plot(x_low, y_low, 'o', color='None', markeredgecolor='b', label='high')
    ax.plot(x_high, y_high, 'x', color='r', label='high')
    ax.plot(x_pred, y_pred, 'g', label=r'$y_{\mathrm{pred}}$')
    ax.plot(x, y, 'k:', label='true')
    ax.legend()
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_xlim(x[0], x[-1])
    ax.grid()
    fig.tight_layout(pad=0)
    return fig


if __name__ == '__main__':
    # Generate data.
    x = torch.linspace(0, 1, 501).reshape(-1, 1)
    y = func_high(x)
    x_low = torch.linspace(0, 1, 51).reshape(-1, 1)
    x_high = torch.linspace(0, 1, 14).reshape(-1, 1)
    y_low = func_low(x_low)
    y_high = func_high(x_high)
    loader_low = torch.utils.data.DataLoader(XYDataSet(x_low, y_low),
                                             batch_size=len(x_low))
    loader_high = torch.utils.data.DataLoader(XYDataSet(x_high, y_high),
                                              batch_size=len(x_low))
    figure1(x_low, y_low, x_high,  y_high, x, y)
    
    model = MFNN(1, 1, [16], [16, 16], [16, 16, 16, 16],  torch.nn.Tanh)
    model_low = model.low
    model_high = model.high
    loss = torch.nn.MSELoss()

    # low-fidelity data
    trainer1 = Trainer(model_low, loss, torch.optim.Adam,
                       lr=1e-2, milestones=[2000, 8000])
    trainer1.logger.setLevel(logging.WARNING)
    for i in range(10000):
        res = trainer1.train(loader_low)
        if i % 100 == 0:
            print(f'niter: {i}', f'avg loss: {res.average:.4e}',
                  f'loss std: {res.std:.4e}', sep=', ')
    y1 = model_low.eval()(x).detach()
    figure2(x_low, y_low, x, y1, x, y)

    # high-fidelity data
    trainer2 = Trainer(model_high, loss, torch.optim.Adam,
                       lr=1e-2, milestones=[2000, 8000],
                       weight_decay=0e-4)
    trainer2.logger.setLevel(logging.WARNING)
    for i in range(10000):
        res = trainer2.train(loader_high)
        if i % 100 == 0:
            print(f'niter: {i}', f'avg loss: {res.average:.4e}',
                  f'loss std: {res.std:.4e}', sep=', ')
    y3 = model_high.eval()(x).detach()
    figure3(x_low, y_low, x_high, y_high, x, y3, x, y)

    plt.show()
