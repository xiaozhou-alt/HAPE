import torch


def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY

        # HMA / CPE 参数使用更高学习率
        if cfg.MODEL.HMA_ENABLE and cfg.MODEL.FROZEN:
            if any(x in key for x in ['hma', 'cpe', 'trainable', 'adapter']):
                lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.HMA_LR_FACTOR
                print(f"Using {cfg.SOLVER.HMA_LR_FACTOR}x lr for HMA/CPE param: {key}")

        # 偏置通常不衰减
        if "bias" in key:
            weight_decay = 0.0

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    # 选择优化器（用户配置中使用 Adam，所以不会进入 SGD 分支）
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        # 如果使用 SGD，需要 momentum（提供一个默认值）
        momentum = getattr(cfg.SOLVER, 'MOMENTUM', 0.9)
        optimizer = torch.optim.SGD(
            params,
            lr=cfg.SOLVER.BASE_LR,
            momentum=momentum,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )
    else:
        # Adam 或其他优化器
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(
            params,
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY
        )

    return optimizer