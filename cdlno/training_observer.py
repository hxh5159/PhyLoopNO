"""Small epoch policy/observer. Task entry integrations belong to later stages."""
from __future__ import annotations

from dataclasses import dataclass
import logging

from .training_state import TASKS, ResumeError, isolated_evaluation


@dataclass(frozen=True)
class EpochPolicy:
    task: str
    total_epochs: int
    visualize_every: int = 50
    checkpoint_every: int | None = None

    def __post_init__(self):
        if self.task not in TASKS:
            raise ValueError(f'unknown task: {self.task}')
        if type(self.total_epochs) is not int or self.total_epochs < 1:
            raise ValueError('total_epochs must be a positive integer')
        if self.checkpoint_every is None:
            object.__setattr__(self, 'checkpoint_every', 100 if self.task in ('ns', 'car', 'airfrans', 'airfoil') else 0)
        for value in (self.visualize_every, self.checkpoint_every):
            if type(value) is not int or value < 0:
                raise ValueError('intervals must be nonnegative integers')

    def events(self, completed_epoch):
        if type(completed_epoch) is not int or not 1 <= completed_epoch <= self.total_epochs:
            raise ValueError('completed_epoch must be in [1,total_epochs]')
        final = completed_epoch == self.total_epochs
        return dict(visualize=bool(self.visualize_every and (final or completed_epoch % self.visualize_every == 0)),
                    checkpoint=final or bool(self.checkpoint_every and completed_epoch % self.checkpoint_every == 0),
                    final=final)


class EpochObserver:
    """Call after original epoch training/validation; owns no optimizer loop.

    Visualization callbacks receive epoch only, close over task inputs, and
    must not update parameters. Errors are returned and warned; saving still
    runs. Archive write errors propagate. Saved history is supplied by caller.
    """
    def __init__(self, archive, policy: EpochPolicy):
        if (archive.protocol.task, archive.protocol.total_epochs) != (policy.task, policy.total_epochs):
            raise ResumeError('observer policy differs from training protocol')
        self.archive, self.policy = archive, policy
        self._last_completed = 0

    def after_epoch(self, completed_epoch, model, optimizer, scheduler, *,
                    global_step, scheduler_steps, normalizers, visualize=None,
                    generators=None, scaler=None, history=None, extra=None):
        events = self.policy.events(completed_epoch)
        if completed_epoch <= self._last_completed:
            raise ResumeError('observer epoch already processed or out of order')
        result = dict(completed_epoch=completed_epoch, events=events, visualization_error=None, checkpoint=None)
        if events['visualize'] and visualize is not None:
            try:
                with isolated_evaluation(model, generators=generators):
                    visualize(completed_epoch)
            except Exception as e:
                result['visualization_error'] = f'{type(e).__name__}: {e}'
                logging.getLogger(__name__).warning('Training visualization failed: %s', result['visualization_error'])
        if events['checkpoint']:
            result['checkpoint'] = self.archive.save(model, optimizer, scheduler,
                completed_epochs=completed_epoch, global_step=global_step, scheduler_steps=scheduler_steps,
                normalizers=normalizers, generators=generators, scaler=scaler, history=history, extra=extra)
        self._last_completed = completed_epoch
        return result
