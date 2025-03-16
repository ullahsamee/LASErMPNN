import torch


def get_std_opt(parameters, d_model, step, warmup_steps=4000, factor=2):
    """
    Just return the default optimizer as ripped from the attention is all you need paper.
    Uses warmup_steps = 4000
    """
    return NoamOpt(
        d_model, factor, warmup_steps, torch.optim.Adam(parameters, lr=0, betas=(0.9, 0.98), eps=1e-9), step
    )


class NoamOpt:
    """
    Optimizer wrapper that implements rate.
    This comes from the Attention is All You Need paper:
        'increases the learning rate linearly for the first warmup_steps training steps, 
        and decreases it thereafter proportionally to the inverse square root of the step number. 
    """
    def __init__(self, model_size, factor, warmup, optimizer, step):
        self.optimizer = optimizer
        self._step = step
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    @property
    def param_groups(self):
        """Return param_groups."""
        return self.optimizer.param_groups

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * (self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self, **kwargs):
        self.optimizer.zero_grad(**kwargs)
