from tabnanny import verbose
from typing import Any

import torch
import yacs.config


def create_scheduler(config: yacs.config.CfgNode, optimizer: Any, train_loader) -> Any:
    if config.scheduler.type == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=config.scheduler.milestones,
            gamma=config.scheduler.lr_decay)
    elif config.scheduler.type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.scheduler.epochs,
            eta_min=config.scheduler.lr_min_factor)
    elif config.scheduler.type == 'onecycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.scheduler.max_lr,
            pct_start=config.scheduler.pct_start,
            epochs=config.scheduler.epochs,
            steps_per_epoch=len(train_loader),
            div_factor=config.scheduler.div_factor,
            final_div_factor=config.scheduler.final_div_factor,
            three_phase=True,
            anneal_strategy='linear',
            verbose=config.scheduler.verbose,
        )
    else:
        raise ValueError()
    return scheduler
