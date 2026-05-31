import torch
from torch import nn
from torch.nn import functional as F

from .block_hma import ResidualAttentionBlockWithHMA
# 注意：你的文件夹名是 componet_module，保持一致
from .componet_module import LayerNorm, QuickGELU
from modeling.backbones.vit_pytorch import trunc_normal_


class Transformer(nn.Module):
    """
    完整的HMA增强ViT模型，和原DeMo的ViT输入输出100%兼容
    参考原 model.py 中的 VisionTransformer 类实现
    输入：RGB/NI/TI 三个模态的原始图像 [B, 3, 256, 128]
    输出：三个模态的token序列 [B, 129, 768]，和原代码完全一致
    """
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, cfg=None):
        super().__init__()

        self.width = width  # 768
        self.layers = layers  # 12
        self.heads = heads  # 12
        self.cfg = cfg

        # ===================== 1. 参考原 model.py 的 VisionTransformer，添加完整预处理层 =====================
        # Patch Embedding (Conv1)
        self.conv1 = nn.Conv2d(
            in_channels=3, 
            out_channels=width, 
            kernel_size=16,  # ViT-B/16的patch size
            stride=16,       # 和原代码一致
            bias=False
        )
        
        # CLS Token
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        
        # 位置编码 (256/16=16, 128/16=8 → 16*8+1=129个位置)
        self.positional_embedding = nn.Parameter(scale * torch.randn(16 * 8 + 1, width))
        
        # Pre-LayerNorm
        self.ln_pre = LayerNorm(width)

        # ===================== 2. 共享TCA模块 =====================
        self.trainable_tca = nn.Sequential(
            nn.Linear(self.width, int(self.width * 2)),
            nn.GELU(),
            nn.Linear(int(self.width * 2), self.width),
        )

        # ===================== 3. 集成HMA的Transformer Block =====================
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlockWithHMA(width, heads, attn_mask, index=i, shared=self.trainable_tca)
            for i in range(layers)
        ])

        # ===================== 4. 早期融合占位（保持原代码结构） =====================
        self.trainable_early_fusion = nn.Sequential(*[
            Nothing(self.width) for _ in range(layers)
        ])

        # 初始化权重
        trunc_normal_(self.positional_embedding, std=.02)
        trunc_normal_(self.class_embedding, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=.001)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def pre(self, x, cv_emb):
        """
        参考原 model.py 中的 VisionTransformer.pre() 函数
        把原始图像转换成token序列
        :param x: 原始图像 [B, 3, 256, 128]
        :param cv_emb: 相机嵌入
        :return: token序列 [L, B, D] (L=129, D=768)
        """
        # 1. Patch Embedding
        x = self.conv1(x)  # [B, 768, 16, 8]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, 768, 128]
        x = x.permute(0, 2, 1)  # [B, 128, 768]
        
        # 2. 拼接CLS Token
        class_emb = self.class_embedding.to(x.dtype).to(x.device)
        class_emb = class_emb.unsqueeze(0).repeat(x.shape[0], 1, 1)  # [B, 1, 768]
        x = torch.cat([class_emb, x], dim=1)  # [B, 129, 768]
        
        # 3. 相机嵌入注入（兼容原代码）
        if cv_emb is not None:
            x[:, 0] = x[:, 0] + cv_emb.squeeze(1)
        
        # 4. 位置编码
        x = x + self.positional_embedding.to(x.dtype).to(x.device)
        
        # 5. Pre-LayerNorm
        x = self.ln_pre(x)
        
        # 6. 转置成 [L, B, D]，和原代码的输入格式完全一致
        x = x.permute(1, 0, 2)  # [129, B, 768]
        return x

    def forward(self, RGB: torch.Tensor, NI: torch.Tensor, TI: torch.Tensor, cv_embed=None, modality=None):
        """
        和原DeMo代码的forward接口100%兼容
        :param RGB: RGB模态图像 [B, 3, 256, 128]
        :param NI: NIR模态图像 [B, 3, 256, 128]
        :param TI: TIR模态图像 [B, 3, 256, 128]
        :param cv_embed: 相机嵌入（兼容原代码）
        :param modality: 模态标识（兼容原代码）
        :return: RGB/NI/TI 三个模态的token序列 [129, B, 768]
        """
        # 第一步：对每个模态做图像预处理
        RGB = self.pre(RGB, cv_embed)
        NI = self.pre(NI, cv_embed)
        TI = self.pre(TI, cv_embed)

        for i in range(self.layers):
            # 早期融合
            RGB, NI, TI = self.trainable_early_fusion[i](RGB, NI, TI)
            # 处理当前层的各模态
            RGB = self.resblocks[i](RGB, modality='rgb', index=i)
            NI = self.resblocks[i](NI, modality='nir', index=i)
            TI = self.resblocks[i](TI, modality='tir', index=i)
            
        RGB = RGB.permute(1, 0, 2)  # [129, B, 768] -> [B, 129, 768]
        NI = NI.permute(1, 0, 2)
        TI = TI.permute(1, 0, 2)

        # 输出和原代码完全一致，后续逻辑无需任何修改
        return RGB, NI, TI


class Nothing(nn.Module):
    def __init__(self, width):
        super().__init__()

    def forward(self, RGB, NIR, TIR):
        return RGB, NIR, TIR