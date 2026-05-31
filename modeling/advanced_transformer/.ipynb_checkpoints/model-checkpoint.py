import torch
from torch import nn
import torch.nn.functional as F
from modeling.advanced_transformer.componet_module import LayerNorm
from .advanced_transformer_hma import Transformer

class VisionTransformer(nn.Module):
    def __init__(self, h_resolution: int, w_resolution: int, patch_size: int, stride_size: int, width: int, layers: int,
                 heads: int, output_dim: int, cfg: dict):
        super().__init__()
        self.h_resolution = h_resolution
        self.w_resolution = w_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=stride_size,
                               bias=False)

        scale = width ** -0.5

        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(h_resolution * w_resolution + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, cfg=cfg)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    def forward(self, RGB: torch.Tensor, NI: torch.Tensor, TI: torch.Tensor, cv_emb=None, modality=None):

        RGB = self.pre(RGB, cv_emb)
        NI = self.pre(NI, cv_emb)
        TI = self.pre(TI, cv_emb)

        RGB, NI, TI = self.transformer(RGB, NI, TI)

        RGBproj = self.post(RGB, modality='inter')
        NIproj = self.post(NI, modality='inter')
        TIproj = self.post(TI, modality='inter')


        return RGBproj, NIproj, TIproj

    def pre(self, x, cv_emb):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat(
            [self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
             x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        if cv_emb != None:
            x[:, 0] = x[:, 0] + cv_emb.squeeze(1)
        x = x + self.positional_embedding.to(x.dtype)

        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        return x

    def post(self, x, modality=None):
        x = x.permute(1, 0, 2)  # LND -> NLD
        # x = self.ln_post(x)
        xproj = x # @ self.proj

        return xproj

def convert_weights(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.float()
            if l.bias is not None:
                l.bias.data = l.bias.data.float()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.float()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.float()

    model.apply(_convert_weights_to_fp16)

def build_model(cfg, state_dict: dict, h_resolution: int, w_resolution: int, vision_stride_size: int):
    
    vision_width = state_dict["visual.conv1.weight"].shape[0]
    vision_layers = len(
        [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
    vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
    grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
    image_resolution = vision_patch_size * grid_size
        
    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]  # 77 (77,512)
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

    from .state_dict_mapping.clip_vit_b_16 import convert_state_dict
    state_dict = convert_state_dict(state_dict)

    model = VisionTransformer(
            h_resolution=h_resolution,
            w_resolution=w_resolution,
            patch_size=vision_patch_size,
            stride_size=vision_stride_size,
            width=vision_width,
            layers=vision_layers,
            heads= vision_width // 64,
            output_dim=embed_dim,
            cfg=cfg
        )

    state_dict["positional_embedding"] = resize_pos_embed(state_dict["positional_embedding"],
                                                                 model.positional_embedding, h_resolution,
                                                                 w_resolution,cfg)

    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)

    try:
        print(f"Successfully load ckpt!")
        incompatibleKeys = model.load_state_dict(state_dict, strict=False)
        print(incompatibleKeys)
    except Exception as e:
        print(f"Failed loading checkpoint!")
    return model.eval()


def load_vit(cfg, backbone_name, h_resolution, w_resolution, vision_stride_size):
    model_path = cfg.MODEL.PRETRAIN_PATH_T

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = build_model(cfg,state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model

import math


def resize_pos_embed(posemb, posemb_new, hight, width,cfg=None):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224

    print('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)

    ntok_new = posemb_new.shape[0]  # 129,2048

    posemb_token, posemb_grid = posemb[:1], posemb[1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))  # 14
    print('Position embedding resize to height:{} width: {}'.format(hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid.squeeze()], dim=0)
    return posemb

if __name__ == '__main__':
    pass