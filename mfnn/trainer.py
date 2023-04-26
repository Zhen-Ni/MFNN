#!/usr/bin/env python3

from __future__ import annotations
import sys
import time
import pickle
import typing

import tqdm
import torch


from .utils import DEVICE, Statistics, free_memory, copy_to, GatedStdout

__all__ = ('Trainer',)


_VALID_SECOND_TIME_WARNING = True


class Trainer():
    """Class for training a model."""

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 critrion: torch.nn.Module,
                 *,
                 device: torch.device | int | str | None = None,
                 start_epoch: int = 0,
                 filename: str | None = None,
                 scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
                 forced_gc: bool = False,
                 suppress_display: bool = False
                 ):
        self.model = model
        self.optimizer = optimizer
        self.critrion = critrion          # loss function
        # Use property setter to move model and loss function to target device
        self.device = DEVICE if device is None else device

        self.epoch = start_epoch
        self.filename = 'trainer.trainer' if filename is None else filename
        self.scheduler = scheduler
        self.is_forced_gc = forced_gc
        self.stdout = GatedStdout(suppress_display)
        
        self.history: dict[str, list[float]] = {'train_loss': [],
                                                'validate_loss': [],
                                                }

    @property
    def device(self) -> torch.device:
        return self._device

    @device.setter
    def device(self, device: torch.device | int | str) -> Trainer:
        self._device = torch.device(device)
        self.model.to(self._device)
        self.critrion.to(self._device)
        return self

    @property
    def lr(self) -> list[float]:
        return [pg['lr'] for pg in self.optimizer.param_groups]

    @lr.setter
    def lr(self, lr: float):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def save(self, device: torch.device | int | str = "cpu"):
        """Save trainer object.

        The trainer is saved to the given `device`, along with the
        model. The `device` specified here does not change the device
        type of the trainer instance, but only saves all the variables
        to this `device`. The default target device is "cpu".

        The `device` specified in this method has nothing to do with
        the `load` method's `device` argument. The `device` argument
        is introduced here to solve the problem that a cuda model can
        not be saved and then loaded on another computer without
        gpu. So it is always suggested to set the `device` argument
        here to "cpu" to make sure it can be loaded on any computer.
        """
        data = copy_to(self.__dict__, torch.device(device))
        with open(self.filename, 'wb') as f:
            f.write(pickle.dumps((data, self.device)))

    def save_as(self, filename: str):
        self.filename = filename
        return self.save()

    @staticmethod
    def load(filename: str,
             device: torch.device | int | str | None = None
             ) -> Trainer:
        """Load a trainer object.

        Load the trainer to given `device`. Specifying `device`
        argument here would also change the loaded trainer's `device`
        property. If device is not given, it is defaulted to the
        object's `device` property.
        """
        with open(filename, 'rb') as f:
            data, default_device = pickle.loads(f.read())
        if device is None:
            data = copy_to(data, default_device)
        else:
            device = torch.device(device)
            data = copy_to(data, device)
            data['_device'] = device
        res = object.__new__(Trainer)
        res.__dict__.update(data)
        return res

    def train(self,
              loader: torch.utils.data.DataLoader
              ) -> Statistics[float]:
        "Train the model by given dataloader."
        t_start = time.time()
        self.model.train()
        tq = tqdm.tqdm(loader,
                       desc="train",
                       ncols=None,
                       leave=False,
                       file=self.stdout,
                       unit="batch")
        loss_meter: Statistics[float] = Statistics()
        # User defined preprocess
        additional_data = self.additional_train_preprocess(tq)
        for x, y in tq:
            current_batch_size = x.shape[0]
            x = x.to(self.device)
            y = y.to(self.device)

            # compute prediction error
            y_pred = self.model(x)
            loss = self.critrion(y_pred, y)
            # backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # record results
            loss_meter.update(loss.item(), current_batch_size)
            tq.set_postfix(loss=f"{loss_meter.value:.4e}")

            # Do some user-defined process.
            self.additional_train_process(additional_data,
                                          y_pred, y, loss, tq)

            # Free some space before next round.
            del x, y, y_pred, loss
            if self.is_forced_gc:
                free_memory()

        # Save information for this epoch.
        self.epoch += 1
        self.history['train_loss'].append(loss_meter.average)

        # User_defined postprocess.
        self.additional_train_postprocess(additional_data)

        print(f'train result [{self.epoch}]: '
              f'avg loss = {loss_meter.average:.4e}, '
              f'wall time = {time.time()- t_start:.2f}s',
              file=self.stdout)
        if self.scheduler:
            self.scheduler.step()

        return loss_meter

    def validate(self,
                 loader: torch.utils.data.DataLoader,
                 ) -> Statistics[float]:
        """Validate the model."""
        t_start = time.time()
        self.model.eval()
        tq = tqdm.tqdm(loader,
                       desc="valid",
                       ncols=None,
                       leave=False,
                       file=self.stdout,
                       unit="batch")
        loss_meter: Statistics[float] = Statistics()
        # User-defined preprocess
        additional_data = self.additional_validate_preprocess(tq)

        for x, y in tq:
            current_batch_size = x.size(0)
            x = x.to(self.device)
            y = y.to(self.device)

            with torch.no_grad():
                y_pred = self.model(x)
            loss = self.critrion(y_pred, y)
            loss_meter.update(loss.item(), current_batch_size)
            tq.set_postfix(loss=f"{loss_meter.value:.4e}")

            # Do some user-defined process.
            self.additional_validate_process(additional_data,
                                             y_pred, y, loss, tq)

            del x, y, y_pred, loss
            if self.is_forced_gc:
                free_memory()

        # Save validation results only the fisrt run.
        if len(self.history['validate_loss']) < self.epoch:
            self.history['validate_loss'].append(loss_meter.average)
        else:
            global _VALID_SECOND_TIME_WARNING
            if _VALID_SECOND_TIME_WARNING:
                sys.stderr.write("The model is validated for the "
                                 "second time in the same epoch, "
                                 "validation result will not be "
                                 "recorded. "
                                 "This warning will be "
                                 "turned off in this session.\n")
                _VALID_SECOND_TIME_WARNING = False

        # User_defined postprocess.
        self.additional_validate_postprocess(additional_data)

        print(f'valid result [{self.epoch}]: '
              f'avg loss = {loss_meter.average:.4e}, '
              f'wall time = {time.time()- t_start:.2f}s',
              file=self.stdout)
        return loss_meter

    def step(self,
             train_dataloader: torch.utils.data.DataLoader,
             valid_dataloader: torch.utils.data.DataLoader,
             save_trainer: bool = True,
             save_best_model: str | None = None,
             ) -> bool:
        """Train and validate the model, return if it is the best model.

        Note that if `save_best_model` is enabled, it selects the best
        model by its validation loss, which might not be good for
        classification problems.

        Parameters
        ----------
        save_trainer: bool, optional
            whether to save the trainer after this epoch. defaults to
            True.
        save_best_model: str, optional
            Filename for saving the model if it is the best model
            indicated by the validation procedure. If not given, the
            model will not be saved automatically.

            """
        print(f'    ---- Epoch {self.epoch} ----    ', file=self.stdout)
        self.train(train_dataloader)
        loss = self.validate(valid_dataloader)
        if save_trainer:
            self.save()
        if loss == min(self.history['validate_loss']):
            if save_best_model:
                print('This model will be saved as the best model.',
                      file=self.stdout)
                with open(save_best_model, 'wb') as f:
                    torch.save(self.model, f)
            return True
        return False

    # Class methods can be overwritten.

    def additional_train_preprocess(self, tq: tqdm.std.tqdm) -> typing.Any:
        """Additional pre-process in each epoch.

        This method can be overwritten to do some additional work
        before iterations in each epoch (eg, prepare dict for
        `additional_train_process`). The return value of the method
        will be used for `additional_train_process` and
        `additional_train_postprocess`

        Parameters
        ----------
        tq: tqdm object
            The tqdm object. Can modify display here.    

        Return
        ------
        additional_data: typing.Any
            The returned data will be used processed in each batch
            and the end of the epoch.
        """
        return None

    def additional_train_process(self,
                                 additional_data: typing.Any,
                                 y_pred: torch.Tensor,
                                 y_true: torch.Tensor,
                                 loss: torch.Tensor,
                                 tq: tqdm.std.tqdm):
        """Additional process after training the model in each batch.

        This method can be overwritten to do some additional work
        after the loss is calculated in each batch (eg, calculate
        top-k error in classification task). This function has no
        return value.

        Parameters
        ----------
        additional_data: typing.Any
            Defined in `additional_train_preprocess`.
        y_pred: torch.Tensor
            Predicted value by model.
        y_true: torch.Tensor
            True value given by train dataset.
        loss: torch.Tensor
            Loss of this batch given by critrion.
        tq: tqdm object
            Can modify display here.
        """
        pass

    def additional_train_postprocess(self,
                                     additional_data: typing.Any):
        """Additional postprocess in each epoch.

        This method can be overwritten to do some additional work
        after the epoch is finished (eg. saving `additional_data`).

        Parameters
        ----------
        additional_data: typing.Any
            Defined in `additional_train_preprocess`.
        """
        pass

    def additional_validate_preprocess(self, tq: tqdm.std.tqdm) -> typing.Any:
        """Additional pre-process in each epoch.

        This method can be overwritten to do some additional work
        before iterations in each epoch (eg, prepare dict for
        `additional_validate_process`). The return value of the method
        will be used for `additional_validate_process` and
        `additional_validate_postprocess`

        Parameters
        ----------
        tq: tqdm object
            The tqdm object. Can modify display here.    

        Return
        ------
        additional_data: typing.Any
            The returned data will be used processed in each batch
            and the end of the epoch.
        """
        return None

    def additional_validate_process(self,
                                    additional_data: typing.Any,
                                    y_pred: torch.Tensor,
                                    y_true: torch.Tensor,
                                    loss: torch.Tensor,
                                    tq: tqdm.std.tqdm):
        """Additional process after model validation in each batch.

        This method can be overwritten to do some additional work
        after the loss is calculated in each batch (eg, calculate
        top-k error in classification task). This function has no
        return value.

        Parameters
        ----------
        additional_data: typing.Any
            Defined in `additional_validate_preprocess`.
        y_pred: torch.Tensor
            Predicted value by model.
        y_true: torch.Tensor
            True value given by validation dataset.
        loss: torch.Tensor
            Loss of this batch given by critrion.
        tq: tqdm object
            Can modify display here.
        """
        pass

    def additional_validate_postprocess(self,
                                        additional_data: typing.Any):
        """Additional postprocess in each epoch.

        This method can be overwritten to do some additional work
        after the epoch is finished (eg. saving `additional_data`).

        Parameters
        ----------
        additional_data: typing.Any
            Defined in `additional_validate_preprocess`.
        """
        pass
