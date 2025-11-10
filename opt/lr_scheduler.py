import torch
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.scheduler import Scheduler
import torch.optim.lr_scheduler as lr_sche


class LinearLRScheduler(Scheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 lr_min_rate: float,
                 warmup_t=0,
                 warmup_lr_init=0.,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True,
                 ) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        self.t_initial = t_initial
        self.lr_min_rate = lr_min_rate
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            t = t - self.warmup_t
            total_t = self.t_initial - self.warmup_t
            lrs = [v - ((v - v * self.lr_min_rate) * (t / total_t)) for v in self.base_values]
        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None


def build_scheduler(args, optimizer, n_iter_per_epoch):
    num_steps = int(args.epochs * n_iter_per_epoch)
    warmup_steps = int(args.warmup_epochs * n_iter_per_epoch)
    decay_steps = int(args.decay_epochs * n_iter_per_epoch)
    scheduler_name = args.lr_scheduler.lower()
    match scheduler_name:
        case 'cosine':
            lr_scheduler = CosineLRScheduler(
                optimizer=optimizer,
                t_initial=num_steps,
                lr_min=args.min_lr,
                cycle_limit=1,
                warmup_t=warmup_steps,
                warmup_lr_init=args.warmup_lr_init,
                t_in_epochs=False
            )
        case 'linear':
            lr_scheduler = LinearLRScheduler(
                optimizer=optimizer,
                t_initial=num_steps,
                lr_min_rate=0.01,
                warmup_t=warmup_steps,
                warmup_lr_init=args.warmup_lr_init,
                t_in_epochs=False
            )
        case 'step':
            lr_scheduler = StepLRScheduler(
                optimizer=optimizer,
                decay_t=decay_steps,
                decay_rate=args.decay_rate,
                warmup_t=warmup_steps,
                warmup_lr_init=args.warmup_lr_init,
                t_in_epochs=False
            )
        case 'cycle':
            lr_scheduler = lr_sche.OneCycleLR(
                optimizer=optimizer,
                max_lr=args.lr,
                total_steps=num_steps
            )
        case _:
            lr_scheduler = None
            print(f"The {lr_scheduler} lr_scheduler does not exist!")
    return lr_scheduler
