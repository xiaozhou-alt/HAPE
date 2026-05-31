from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F

from modeling.advanced_transformer.componet_module import LayerNorm, QuickGELU
from modeling.backbones.vit_pytorch import trunc_normal_


class HeterogeneousModalityAdapter(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)

        # RGB: 大瓶颈 + SiLU 门控
        self.rgb_down_proj = nn.Linear(d_model, d_model // 2)
        self.rgb_gate_proj = nn.Linear(d_model, d_model // 2)
        self.rgb_gate_act = nn.SiLU()
        self.rgb_up_proj = nn.Linear(d_model // 2, d_model)
        self.rgb_act = nn.SiLU()

        # NIR: 中瓶颈 + Sigmoid 门控 (更稳定)
        self.nir_down_proj = nn.Linear(d_model, d_model // 2)
        self.nir_gate_proj = nn.Linear(d_model, d_model // 2)
        self.nir_gate_act = nn.Sigmoid()
        self.nir_up_proj = nn.Linear(d_model // 2, d_model)
        self.nir_act = nn.ReLU(inplace=True)

        # TIR: 小瓶颈 + 无门控 + GELU
        self.tir_down_proj = nn.Linear(d_model, d_model // 2)
        self.tir_up_proj = nn.Linear(d_model // 2, d_model)
        self.tir_act = nn.GELU()

        # 初始化标准差增大
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        # LayerNorm 等保持不变

    def forward(self, x_patch: torch.Tensor, modality: str):
        if modality == 'rgb':
            gate = self.rgb_gate_act(self.rgb_gate_proj(x_patch))
            x = self.rgb_act(self.rgb_down_proj(x_patch))
            x = gate * x
            x = self.dropout(x)
            x = self.rgb_up_proj(x)
        elif modality == 'nir':
            gate = self.nir_gate_act(self.nir_gate_proj(x_patch))
            x = self.nir_act(self.nir_down_proj(x_patch))
            x = gate * x
            x = self.dropout(x)
            x = self.nir_up_proj(x)
        elif modality == 'tir':
            x = self.tir_act(self.tir_down_proj(x_patch))
            x = self.dropout(x)
            x = self.tir_up_proj(x)
        return x


class ResidualAttentionBlockWithHMA(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, index=None, shared=None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        # HMA核心模块
        self.hma_adapter = HeterogeneousModalityAdapter(d_model=d_model)
        self.trainable_shared_tca = shared

        # 异构CLS Token调优
        self.trainable_rgb_cls_tuning_mhsa = nn.Parameter(torch.zeros(1, 1, d_model))
        self.trainable_rgb_cls_tuning_ffn = nn.Parameter(torch.zeros(1, 1, d_model))
        self.trainable_nir_cls_tuning_mhsa = nn.Parameter(torch.zeros(1, 1, d_model))
        self.trainable_nir_cls_tuning_ffn = nn.Parameter(torch.zeros(1, 1, d_model))
        self.trainable_tir_cls_tuning_mhsa = nn.Parameter(torch.zeros(1, 1, d_model))
        self.trainable_tir_cls_tuning_ffn = nn.Parameter(torch.zeros(1, 1, d_model))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv1d):
            trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def adapter_ffn(self, x: torch.Tensor, modality: str):
        x_cls = x[0].unsqueeze(0)
        x_patch = x[1:]
        
        # TCA 应用于所有模态
        if self.trainable_shared_tca is not None:
            adapter_cls = self.trainable_shared_tca(x_cls)
        else:
            adapter_cls = torch.zeros_like(x_cls)
        
        adapter_patch = self.hma_adapter(x_patch, modality)
        adapter_x = torch.cat((adapter_cls, adapter_patch), dim=0)
        return adapter_x

    def forward(self, x: torch.Tensor, modality=None, index=0):
        batch_size = x.shape[1]
        
        if modality == 'rgb':
            cls_tuning_mhsa = self.trainable_rgb_cls_tuning_mhsa.repeat(1, batch_size, 1)
            cls_tuning_ffn = self.trainable_rgb_cls_tuning_ffn.repeat(1, batch_size, 1)
        elif modality == 'nir':
            cls_tuning_mhsa = self.trainable_nir_cls_tuning_mhsa.repeat(1, batch_size, 1)
            cls_tuning_ffn = self.trainable_nir_cls_tuning_ffn.repeat(1, batch_size, 1)
        elif modality == 'tir':
            cls_tuning_mhsa = self.trainable_tir_cls_tuning_mhsa.repeat(1, batch_size, 1)
            cls_tuning_ffn = self.trainable_tir_cls_tuning_ffn.repeat(1, batch_size, 1)
        else:
            cls_tuning_mhsa = torch.zeros(1, batch_size, x.shape[2], device=x.device, dtype=x.dtype)
            cls_tuning_ffn = torch.zeros(1, batch_size, x.shape[2], device=x.device, dtype=x.dtype)

        # MHSA
        x = x + self.attention(self.ln_1(x) + torch.cat((cls_tuning_mhsa, torch.zeros_like(x[1:])), dim=0))

        # Adapter
        adapter_x = self.adapter_ffn(x, modality)
        
        # MLP
        x = x + self.mlp(self.ln_2(x) + torch.cat((cls_tuning_ffn, torch.zeros_like(x[1:])), dim=0)) + adapter_x
        
        return x