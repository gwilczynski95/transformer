from functools import partial

import torch


def transformer_scheduler(step, d_model, warmup_steps):
    step += 1
    min_val = min(
        step ** (-1 / 2),
        step * warmup_steps ** (-3 / 2)
    )
    return d_model ** (-1 / 2) * min_val


def get_optimizer_and_scheduler(model, d_model, warmup_steps, lr, betas, eps):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas, eps=eps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=partial(
            transformer_scheduler,
            d_model=d_model,
            warmup_steps=warmup_steps
        )
    )
    return optimizer, scheduler
