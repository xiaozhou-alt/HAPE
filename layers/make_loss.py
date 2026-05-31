# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch.nn.functional as F
from .softmax_loss import CrossEntropyLabelSmooth
from .triplet_loss import TripletLoss


def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER

    # Triplet loss
    triplet = TripletLoss(cfg.SOLVER.MARGIN)   # 使用 margin
    print("using triplet loss with margin: {}".format(cfg.SOLVER.MARGIN))

    # Label smoothing
    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)
    else:
        xent = F.cross_entropy

    if sampler == 'softmax':
        def loss_func(score, feat, target, target_cam):
            return xent(score, target)

    elif sampler == 'softmax_triplet':
        def loss_func(score, feat, target, target_cam):
            id_loss = xent(score, target)
            tri_loss = triplet(feat, target)[0]
            return cfg.MODEL.ID_LOSS_WEIGHT * id_loss + cfg.MODEL.TRIPLET_LOSS_WEIGHT * tri_loss
    else:
        raise ValueError(f"Unsupported sampler: {sampler}")

    return loss_func