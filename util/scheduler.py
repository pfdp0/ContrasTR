import warnings
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer


class StepLRWithWarmup(_LRScheduler):
    """Decays the learning rate of each parameter group by gamma every drop_every steps.
    Additionally, it linearly increases the learning rate from 0 to the initial lr within the first warmup_steps steps.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        drop_every (int): Period of learning rate decay.
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.
        last_step (int): The index of last epoch. Default: -1.
        warmup_steps (int): The number of steps for the warmup stage.
            Default: 0.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

    Example:
        >>> # xdoctest: +SKIP
        >>> scheduler = StepLRWithWarmup(optimizer, drop_every=1000, gamma=0.1, warmup_steps=200)
        >>> for epoch in range(15):
        >>>     for step in range(100):
        >>>         ...
        >>>         optimizer.step()
        >>>         scheduler.step()
    """

    def __init__(self, optimizer: Optimizer, drop_every: int, gamma: float = 0.1, last_step: int = -1,
                 warmup_steps: int = 0, verbose: bool = False):
        assert warmup_steps >= 0, 'warmup_steps must be non-negative'
        assert warmup_steps < drop_every, 'warmup_steps must be smaller than drop_every'

        self.drop_every = drop_every
        self.gamma = gamma
        self.warmup_steps = warmup_steps

        super(StepLRWithWarmup, self).__init__(optimizer, last_step, verbose)

    def get_lr(self):
        last_step = self.last_epoch
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)
        if (last_step + 1) <= self.warmup_steps:  # linear warmup stage
            return [base_lr * (last_step + 1) / self.warmup_steps for base_lr in self.base_lrs]
        elif (last_step == 0) or (last_step % self.drop_every != 0):
            return [group['lr'] for group in self.optimizer.param_groups]
        else:
            return [group['lr'] * self.gamma for group in self.optimizer.param_groups]  # decay stage

    def _get_closed_form_lr(self):
        last_step = self.last_epoch
        print(f"Getting closed form lr for step {last_step}")
        print("Base lrs: ", self.base_lrs)
        if (last_step + 1) < self.warmup_steps:
            return [base_lr * (last_step + 1) / self.warmup_steps for base_lr in self.base_lrs]
        else:
            print("After warmup: ", [base_lr * self.gamma ** (last_step // self.drop_every) for base_lr in self.base_lrs])
            return [
                base_lr * self.gamma ** (last_step // self.drop_every)
                for base_lr in self.base_lrs
            ]


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from torch.optim import SGD

    model = torch.nn.Linear(2, 1, bias=True)
    param_groups = [{'params': model.weight, 'lr': 1.0}, {'params': model.bias, 'lr': 2.0}]
    optimizer = SGD(param_groups, lr=1.0)
    scheduler = StepLRWithWarmup(optimizer, drop_every=100, gamma=0.1, warmup_steps=10)

    lrs = []
    lrs2 = []
    lrs_verif = []
    lrs2_verif = []
    for epoch in range(20):
        for step in range(10):
            ...
            lr_v1, lr_v2 = scheduler._get_closed_form_lr()
            lrs.append(optimizer.param_groups[0]['lr'])
            lrs2.append(optimizer.param_groups[1]['lr'])

            lrs_verif.append(lr_v1)
            lrs2_verif.append(lr_v2)
            optimizer.step()
            scheduler.step()


    plt.plot(lrs)
    plt.plot(lrs2)
    plt.plot(lrs_verif, linestyle='--')
    plt.plot(lrs2_verif, linestyle='--')
    plt.legend(['lr1', 'lr2', 'lr1_verif', 'lr2_verif'])
    plt.xlabel('Step')
    plt.ylabel('Learning rate')
    plt.ylim(0, 3)
    plt.show()
