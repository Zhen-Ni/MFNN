#!/usr/bin/env python3

from __future__ import annotations
import typing
from io import BytesIO
import time
import logging
import pickle
import torch

__all__ = ('device', 'Trainer')


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def copy_to(data, device: torch.device | int | str | None = None):
    """Copy data to device without unnecessary copy."""
    # Avoid useless copy in gpu.
    # See https://discuss.pytorch.org/t/how-to-make-a-copy-of-a-gpu-model-on-the-cpu/90955/4
    if device is None:
        return data
    memory = BytesIO()
    torch.save(data, memory, pickle_protocol=-1)
    memory.seek(0)
    data = torch.load(memory, map_location=device)
    memory.close()
    return data


def get_logger(name: str) -> logging.Logger:
    "Get a logger with given name."
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    # formatter = logging.Formatter('%(asctime)s %(message)s')
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    return logger


T = typing.TypeVar('T', bound=float)


class Statistics(typing.Generic[T]):
    """Get statistical data like sum, average, etc of the data."""
    __slots__ = '_count', '_value', '_s1', '_s2'

    def __init__(self):
        self._count = 0
        self._value: T = 0.
        self._s1: float = 0.            # sum of sample values
        self._s2: float = 0.            # sum of squared sample values

    def update(self, value: T, n: int = 1) -> Statistics[T]:
        self._count += n
        self._value = value
        self._s1 = self.value * n
        self._s2 += value ** 2 * n
        return self

    @property
    def count(self) -> int:
        return self._count

    @property
    def value(self) -> T:
        return self._value

    @property
    def sum(self) -> float:
        return self._s1

    @property
    def average(self) -> float:
        return self._s1 / self._count

    @property
    def variance(self) -> float:
        return self._s2 / self._count - self.average ** 2

    @property
    def std(self) -> float:
        return self.variance ** .5


class Trainer():
    """Class for training a model."""

    def __init__(self,
                 model: torch.nn.Module,
                 loss: torch.nn.Module,
                 optimizer: type[torch.optim.Optimizer] | None = None,
                 *,
                 start_epoch: int = 0,
                 filename: str | None = None,
                 logger: str | None = None,
                 milestones: list[int] = [],
                 gamma: float = 0.1,
                 **kwargs
                 ):
        self.model = model
        self.loss_function = loss.to(self.device)
        self.epoch = start_epoch
        self.filename = 'trainer.trainer' if filename is None else filename
        if isinstance(logger, str):
            self.logger = get_logger(logger)
        else:
            self.logger = get_logger('trainer')

        if optimizer is None:
            optimizer = torch.optim.SGD
        self.optimizer = optimizer(self.model.parameters(),
                                   **kwargs)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=milestones, gamma=gamma,
            last_epoch=self.epoch-1)

        self.history: dict[str, list[float]] = {'train_loss': [],
                                                'train_loss_std': [],
                                                'test_loss': [],
                                                'test_loss_std': []}

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def lr(self) -> list[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]

    @lr.setter
    def lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, device: torch.device | int | str | None = "cpu"):
        data = copy_to(self.__dict__, device)
        with open(self.filename, 'wb') as f:
            f.write(pickle.dumps((data, self.device)))

    def save_as(self, filename: str):
        self.filename = filename
        return self.save()

    @staticmethod
    def load(filename: str,
             device: torch.device | int | str | None = None
             ) -> Trainer:
        with open(filename, 'rb') as f:
            data, default_device = pickle.loads(f.read())
        if device is None:
            data = copy_to(data, default_device)
        else:
            data = copy_to(data, device)
        res = object.__new__(Trainer)
        res.__dict__.update(data)
        res.logger = get_logger(res.logger.name)
        return res

    def train(self,
              train_dataloader: torch.utils.data.DataLoader,
              log_every: int = 50
              ) -> Statistics[float]:
        "Train the model by given dataloader."
        self.logger.info(f'---- Epoch {self.epoch} ----')
        t_start = time.time()
        self.model.train()
        loss_stats: Statistics[float] = Statistics()
        # Make sure whether dataset is Sized
        assert hasattr(train_dataloader.dataset, '__len__'), \
            "dataset should be sized"
        size = len(train_dataloader.dataset)     # type: ignore
        trained_samples = 0
        for i, (x, y) in enumerate(train_dataloader):
            current_batch_size = x.shape[0]
            x = x.to(self.device)
            y = y.to(self.device)
            # compute prediction error
            y_pred = self.model(x)
            loss = self.loss_function(y_pred, y)
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            # record result
            trained_samples += current_batch_size
            loss_stats.update(loss.item(), current_batch_size)
            if log_every and (i + 1) % log_every == 0:
                self.logger.info(f'loss = {loss_stats.value:.7f} '
                                 f'[{trained_samples:>5d}/{size:>5d}, '
                                 f'{trained_samples / size * 100:>0.1f}%]')
        self.logger.info(f'train result: '
                         f'avg loss = {loss_stats.average:.4e}, '
                         f'loss std = {loss_stats.std:.4f}, '
                         f'wall time = {time.time()- t_start:.2f}s')
        self.scheduler.step()
        # Save information for this epoch.
        self.epoch += 1
        self.history['train_loss'].append(loss_stats.average)
        self.history['train_loss_std'].append(loss_stats.std)
        return loss_stats

    def test(self, test_dataloader: torch.utils.data.DataLoader,
             log_every: int = 50
             ) -> Statistics[float]:
        "Test the model using top-k error by given dataloader."
        t_start = time.time()
        self.model.eval()
        loss_stats: Statistics[float] = Statistics()
        # Make sure whether dataset is Sized
        assert hasattr(test_dataloader.dataset, '__len__'), \
            "dataset should be sized"
        size = len(test_dataloader.dataset)     # type: ignore
        tested_samples = 0
        with torch.no_grad():
            for i, (x, y) in enumerate(test_dataloader):
                current_batch_size = x.shape[0]
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.model(x)
                loss_stats.update(self.loss_function(y_pred, y).item(),
                                  current_batch_size)
                if log_every and (i + 1) % log_every == 0:
                    self.logger.info(f'loss = {loss_stats.value:.7f} '
                                     f'[{tested_samples:>5d}/{size:>5d}, '
                                     f'{tested_samples / size * 100:>0.1f}%]')
        self.logger.info(f'test result: '
                         f'avg loss = {loss_stats.average:.4e}, '
                         f'loss std = {loss_stats.std:.4f}, '
                         f'wall time = {time.time()-t_start:.2f}s')
        # Save test results only the fisrt run.
        if len(self.history['test_loss']) < self.epoch:
            self.history['test_loss'].append(loss_stats.average)
            self.history['test_loss_std'].append(loss_stats.average)
        return loss_stats
