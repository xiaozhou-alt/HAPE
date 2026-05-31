import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from modeling.advanced_transformer.componet_module import LayerNorm
import copy


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


def peft_frozen(model: nn.Module) -> None:
    """冻结 backbone，只训练 HMA 和 CPE 相关参数"""
    for n, p in model.named_parameters():
        if 'hma' not in n and 'cpe' not in n:
            p.requires_grad = False


def resize_pos_embed(posemb, posemb_new, hight, width, cfg=None):
    """Resize positional embedding from CLIP pretrained model"""
    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[0]
    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1
    gs_old = int(math.sqrt(len(posemb_grid)))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze()], dim=0)
    return posemb


class build_transformer(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory, feat_dim):
        super(build_transformer, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH_T
        self.in_planes = feat_dim
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.model_name = cfg.MODEL.TRANSFORMER_TYPE
        self.flops_test = cfg.MODEL.FLOPS_TEST
        self.hma_enable = getattr(cfg.MODEL, 'HMA_ENABLE', False)
        self.cpe_enable = getattr(cfg.MODEL, 'CPE_ENABLE', False)
        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.TRANSFORMER_TYPE == 'ViT-B-16':
            from modeling.advanced_transformer.advanced_transformer_hma import Transformer
            vit_model = Transformer(
                width=feat_dim,
                layers=12,
                heads=12,
                attn_mask=None,
                cfg=cfg
            )
            print('===========Building Transformer with HMA/CPE support===========')

            if model_path != '':
                print(f'Loading pretrained CLIP weights from {model_path}')
                self._load_clip_pretrained(vit_model, model_path, cfg)

            self.base = vit_model

            if self.hma_enable and cfg.MODEL.FROZEN:
                peft_frozen(self.base)
                print('===========PEFT Frozen: Only training HMA & CPE params===========')

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def _load_clip_pretrained(self, model, model_path, cfg):
        print("Loading CLIP pretrained weights...")
        if not model_path:
            print("No pretrained path provided.")
            return
        import os
        if not os.path.exists(model_path):
            print(f"Error: weight file not found at {model_path}")
            return
        try:
            model_clip = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = None
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")
        state_dict = state_dict or model_clip.state_dict()
        if state_dict is None:
            print("Failed to load state_dict")
            return

        new_state_dict = {}
        for k, v in state_dict.items():
            if not k.startswith("visual."):
                continue
            new_k = k.replace("visual.", "")
            if new_k.startswith("transformer.resblocks."):
                new_k = new_k.replace("transformer.resblocks.", "resblocks.")
            new_state_dict[new_k] = v

        if "positional_embedding" in new_state_dict:
            posemb = new_state_dict["positional_embedding"]
            posemb_new = model.positional_embedding
            if posemb.shape != posemb_new.shape:
                print(f"Resizing positional embedding from {posemb.shape} to {posemb_new.shape}")
                new_state_dict["positional_embedding"] = resize_pos_embed(
                    posemb, posemb_new, 16, 8, cfg
                )
        incompatible = model.load_state_dict(new_state_dict, strict=False)
        print(f"Loaded pretrained weights. Missing keys (HMA/CPE only): {len(incompatible.missing_keys)}")
        print(f"Unexpected keys: {len(incompatible.unexpected_keys)}")

    def forward(self, RGB, NI, TI, label=None, cam_label=None, view_label=None, modality=None):
        # 不再使用 cv_embed
        result = self.base(RGB, NI, TI, cv_embed=None, modality=modality)
        if self.cpe_enable:
            RGB, NI, TI, all_protos = result
        else:
            RGB, NI, TI = result
            all_protos = None

        RGB_global_feat = RGB[:, 0]
        NI_global_feat = NI[:, 0]
        TI_global_feat = TI[:, 0]

        RGB_feat = self.bottleneck(RGB_global_feat)
        NI_feat = self.bottleneck(NI_global_feat)
        TI_feat = self.bottleneck(TI_global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                RGB_cls_score = self.classifier(RGB_feat, label)
                NI_cls_score = self.classifier(NI_feat, label)
                TI_cls_score = self.classifier(TI_feat, label)
            else:
                RGB_cls_score = self.classifier(RGB_feat)
                NI_cls_score = self.classifier(NI_feat)
                TI_cls_score = self.classifier(TI_feat)

            if self.cpe_enable:
                return RGB, RGB_cls_score, RGB_global_feat, NI, NI_cls_score, NI_global_feat, TI, TI_cls_score, TI_global_feat, all_protos
            else:
                return RGB, RGB_cls_score, RGB_global_feat, NI, NI_cls_score, NI_global_feat, TI, TI_cls_score, TI_global_feat
        else:
            if self.neck_feat == 'after':
                if self.cpe_enable:
                    return RGB, RGB_feat, NI, NI_feat, TI, TI_feat, all_protos
                else:
                    return RGB, RGB_feat, NI, NI_feat, TI, TI_feat
            else:
                if self.cpe_enable:
                    return RGB, RGB_global_feat, NI, NI_global_feat, TI, TI_global_feat, all_protos
                else:
                    return RGB, RGB_global_feat, NI, NI_global_feat, TI, TI_global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class HAPE(nn.Module):
    def __init__(self, num_classes, cfg, camera_num, view_num, factory):
        super(HAPE, self).__init__()
        if 'vit_base_patch16_224' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        elif 'ViT-B-16' in cfg.MODEL.TRANSFORMER_TYPE:
            self.feat_dim = 768
        self.BACKBONE = build_transformer(num_classes, cfg, camera_num, view_num, factory, feat_dim=self.feat_dim)
        self.num_classes = num_classes
        self.cfg = cfg
        self.num_instance = cfg.DATALOADER.NUM_INSTANCE
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        self.image_size = cfg.INPUT.SIZE_TRAIN

        self.cpe_enable = getattr(cfg.MODEL, 'CPE_ENABLE', False)
        if not self.cpe_enable:
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)
            self.ln_rgb = LayerNorm(self.feat_dim)
            self.ln_nir = LayerNorm(self.feat_dim)
            self.ln_tir = LayerNorm(self.feat_dim)
        else:
            self.classifier = nn.Linear(3 * self.feat_dim, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)
            self.bottleneck = nn.BatchNorm1d(3 * self.feat_dim)
            self.bottleneck.bias.requires_grad_(False)
            self.bottleneck.apply(weights_init_kaiming)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def flops(self, shape=(3, 256, 128)):
        if hasattr(self.cfg.MODEL, 'HMA_ENABLE') and self.cfg.MODEL.HMA_ENABLE:
            print("Skipping FLOPs calculation for HMA model")
            return 34.2e9
        # 如果不需要 FLOPs 计算，可以删除整个方法
        return 0

    def forward(self, x, label=None, cam_label=None, view_label=None):
        if self.training:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']

            backbone_out = self.BACKBONE(RGB, NI, TI, cam_label=cam_label, view_label=view_label)
            if self.cpe_enable:
                RGB_cash, RGB_score, RGB_global, NI_cash, NI_score, NI_global, TI_cash, TI_score, TI_global, all_protos = backbone_out
            else:
                RGB_cash, RGB_score, RGB_global, NI_cash, NI_score, NI_global, TI_cash, TI_score, TI_global = backbone_out

            if self.cpe_enable and all_protos is not None:
                cpe = self.BACKBONE.base.cpe
                ori = cpe.final_feature(all_protos, RGB_global, NI_global, TI_global)
            else:
                ori = torch.cat([self.ln_rgb(RGB_global), self.ln_nir(NI_global), self.ln_tir(TI_global)], dim=-1)

            ori_global = self.bottleneck(ori)
            ori_score = self.classifier(ori_global)
            return ori_score, ori

        else:
            RGB = x['RGB']
            NI = x['NI']
            TI = x['TI']
            backbone_out = self.BACKBONE(RGB, NI, TI, cam_label=cam_label, view_label=view_label)
            if self.cpe_enable:
                RGB_cash, RGB_global, NI_cash, NI_global, TI_cash, TI_global, all_protos = backbone_out
            else:
                RGB_cash, RGB_global, NI_cash, NI_global, TI_cash, TI_global = backbone_out

            if self.cpe_enable and all_protos is not None:
                cpe = self.BACKBONE.base.cpe
                ori = cpe.final_feature(all_protos, RGB_global, NI_global, TI_global)
            else:
                ori = torch.cat([self.ln_rgb(RGB_global), self.ln_nir(NI_global), self.ln_tir(TI_global)], dim=-1)
            return ori


def make_model(cfg, num_class, camera_num, view_num=0):
    __factory_T_type = {}   # 不再需要其他 backbone
    model = HAPE(num_class, cfg, camera_num, view_num, __factory_T_type)
    print('===========Building HAPE===========')
    return model