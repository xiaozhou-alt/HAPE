import torch
from torch import nn
from torch.nn import functional as F

from .block_hma import ResidualAttentionBlockWithHMA
from .componet_module import LayerNorm, QuickGELU
from modeling.backbones.vit_pytorch import trunc_normal_


# ========================= CPE 模块 =========================
class CrossModalPrototypeEvolution(nn.Module):
    """
    跨模态原型进化模块 (CPE)
    在每一层后调用，维护跨模态原型 P，并融合最终特征
    """
    def __init__(self, d_model: int, layers: int, cfg):
        super().__init__()
        self.d_model = d_model
        self.layers = layers
        self.cfg = cfg

        # 可学习投影矩阵（用于亲和度计算）
        if cfg.MODEL.CPE_LEARNABLE_AFFINITY:
            self.W_Q = nn.Linear(d_model, d_model, bias=False)
            self.W_K = nn.Linear(d_model, d_model, bias=False)
        else:
            self.W_Q = nn.Identity()
            self.W_K = nn.Identity()

        # 交叉注意力（聚合跨模态 patch）
        if cfg.MODEL.CPE_USE_CROSS_ATTN:
            self.cross_attn = nn.MultiheadAttention(d_model, num_heads=8, batch_first=False)
            self.cross_attn_ln = LayerNorm(d_model)
        else:
            self.cross_attn = None

        # 外来原型注意力池化 query 向量（6个方向）
        self.attn_query = nn.ParameterDict()
        for src in ['R', 'N', 'T']:
            for tgt in ['R', 'N', 'T']:
                if src != tgt:
                    self.attn_query[f"{tgt}_to_{src}"] = nn.Parameter(
                        torch.empty(1, 1, d_model)
                    )
                    nn.init.normal_(self.attn_query[f"{tgt}_to_{src}"], std=0.02)

        # 动态融合 g_R, g_N, g_T
        if cfg.MODEL.CPE_MULTI_MODAL_FUSION == 'transformer':
            self.fusion_transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
                num_layers=1
            )
        elif cfg.MODEL.CPE_MULTI_MODAL_FUSION == 'mlp':
            self.fusion_mlp = nn.Sequential(
                nn.Linear(3 * d_model, d_model * 2),
                nn.GELU(),
                nn.Linear(d_model * 2, d_model)
            )
        else:  # mean
            self.fusion_transformer = None
            self.fusion_mlp = None

        # 用于计算 g 的动态权重 (cR 与 fN→R 的相关性)
        self.dyn_weight_mlp = nn.Sequential(
            nn.Linear(2 * d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 1)
        )

        # 原型进化门控
        if cfg.MODEL.CPE_EVOLVE_TYPE == 'gru':
            self.gru_linear_zr = nn.Linear(2 * d_model, 2 * d_model)
            self.gru_linear_p = nn.Linear(2 * d_model, d_model)
        elif cfg.MODEL.CPE_EVOLVE_TYPE == 'mlp_gate':
            self.gate_mlp = nn.Sequential(
                nn.Linear(2 * d_model, 2),
                nn.Softmax(dim=-1)
            )
        else:  # residual
            self.residual_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            )

        # 原型反馈缩放参数
        if cfg.MODEL.CPE_FEEDBACK_PROTO:
            self.gamma = nn.Parameter(torch.zeros(1))

        # 层级聚合可学习权重
        if cfg.MODEL.CPE_LAYER_AGGR:
            self.layer_score = nn.Parameter(torch.zeros(layers))
        else:
            self.layer_score = None

        # 最终检索特征投影头
        if cfg.MODEL.CPE_RETRIEVAL_FEAT == 'mlp':
            self.feat_proj = nn.Sequential(
                nn.Linear(4 * d_model, d_model * 3),
                nn.GELU(),
                nn.Linear(3 * d_model, 3 * d_model)
            )
        else:
            self.feat_proj = nn.Linear(4 * d_model, 3 * d_model)

        self._init_weights()

    def _init_weights(self):
        pass  # 各模块已经初始化

    def compute_affinity(self, P_X, P_Y):
        """
        计算 X→Y 的余弦相似度矩阵
        P_X, P_Y: [N, B, D]
        返回: [B, N, N]
        """
        Q = self.W_Q(P_X)  # [N, B, D]
        K = self.W_K(P_Y)  # [N, B, D]
        Q = F.normalize(Q, dim=-1)
        K = F.normalize(K, dim=-1)
        # 计算点积
        affinity = torch.einsum('nbd,mbd->bnm', Q, K)  # [B, N, N]
        return affinity

    def aggregate_cross_features(self, P_X, P_Y, affinity):
        """
        从 Y 聚合特征到 X
        若使用交叉注意力，则忽略 affinity
        返回: [N, B, D]
        """
        if self.cross_attn is not None:
            # 交叉注意力：query = P_X, key/value = P_Y
            attn_out, _ = self.cross_attn(P_X, P_Y, P_Y)
            attn_out = self.cross_attn_ln(attn_out + P_X)
            return attn_out
        else:
            # 加权求和
            F_Y_to_X = torch.einsum('bnm,mbd->nbd', affinity, P_Y)  # [N, B, D]
            return F_Y_to_X

    def proto_pooling(self, F_Y_to_X, direction_name):
        if self.cfg.MODEL.CPE_PROTO_POOLING == 'attn':
            q = self.attn_query[direction_name]          # 可能为 [1,1,D] 或 [1,D,1]
            q = q.view(1, 1, -1)                         # 强制变成 [1, 1, D]
            F = F_Y_to_X.permute(1, 0, 2)                # [B, N, D]
            q = q.expand(F.size(0), -1, -1)              # [B, 1, D]
            attn_scores = torch.bmm(F, q.transpose(1, 2)) # [B, N, 1]
            attn_weights = attn_scores.squeeze(-1).softmax(dim=-1)  # [B, N]
            pooled = torch.bmm(attn_weights.unsqueeze(1), F).squeeze(1)  # [B, D]
            return pooled
        else:
            return F_Y_to_X.mean(dim=0)

    def compute_g_and_h(self, f_dict, c_R, c_N, c_T):
        """
        根据外来原型计算 g_R, g_N, g_T 和跨模态摘要 h
        f_dict: {'N→R': ..., 'T→R': ..., ...}
        c_R/c_N/c_T: [B, D]
        """
        B = c_R.size(0)
        # 计算 g_R
        def compute_g(c_x, f1, f2):
            cat1 = torch.cat([c_x, f1], dim=-1)
            cat2 = torch.cat([c_x, f2], dim=-1)
            w1 = torch.sigmoid(self.dyn_weight_mlp(cat1))  # [B, 1]
            w2 = torch.sigmoid(self.dyn_weight_mlp(cat2))
            w_sum = w1 + w2 + 1e-8
            g = (w1 / w_sum) * f1 + (w2 / w_sum) * f2
            return g

        g_R = compute_g(c_R, f_dict['N→R'], f_dict['T→R'])
        g_N = compute_g(c_N, f_dict['R→N'], f_dict['T→N'])
        g_T = compute_g(c_T, f_dict['R→T'], f_dict['N→T'])

        # 融合 g 得到 h
        g_stack = torch.stack([g_R, g_N, g_T], dim=1)  # [B, 3, D]
        if self.cfg.MODEL.CPE_MULTI_MODAL_FUSION == 'transformer':
            h_seq = self.fusion_transformer(g_stack)  # [B, 3, D]
            h = h_seq.mean(dim=1)  # [B, D]
        elif self.cfg.MODEL.CPE_MULTI_MODAL_FUSION == 'mlp':
            g_cat = torch.cat([g_R, g_N, g_T], dim=-1)  # [B, 3D]
            h = self.fusion_mlp(g_cat)
        else:
            h = g_stack.mean(dim=1)  # mean pooling
        return g_R, g_N, g_T, h

    def evolve_proto(self, P_prev, h):
        """
        原型进化，P_prev: [B, D], h: [B, D]
        返回: [B, D]
        """
        if self.cfg.MODEL.CPE_EVOLVE_TYPE == 'gru':
            combined = torch.cat([P_prev, h], dim=-1)  # [B, 2D]
            gates = self.gru_linear_zr(combined)  # [B, 2D]
            z, r = torch.chunk(gates, 2, dim=-1)
            z = torch.sigmoid(z)
            r = torch.sigmoid(r)
            candidate = torch.tanh(self.gru_linear_p(
                torch.cat([r * P_prev, h], dim=-1)
            ))
            P_new = (1 - z) * P_prev + z * candidate
            return P_new
        elif self.cfg.MODEL.CPE_EVOLVE_TYPE == 'mlp_gate':
            combined = torch.cat([P_prev, h], dim=-1)
            alpha = self.gate_mlp(combined)  # [B, 2]
            alpha_self, alpha_ext = alpha[:, 0:1], alpha[:, 1:2]
            P_new = alpha_self * P_prev + alpha_ext * h
            return P_new
        else:  # residual
            P_new = P_prev + self.residual_mlp(h)
            return P_new

    def forward(self, layer_idx, P_prev, RGB_tokens, NI_tokens, TI_tokens, c_R, c_N, c_T):
        """
        :param layer_idx: 当前层索引（0基于）
        :param P_prev: 上一层原型 [B, D]
        :param RGB_tokens, NI_tokens, TI_tokens: [L, B, D] 的全 token 序列
        :param c_R, c_N, c_T: 各模态的 CLS token [B, D]
        :return: 更新后的原型 P_new [B, D]
        """
        # 提取 patch tokens
        P_R = RGB_tokens[1:]  # [N, B, D]
        P_N = NI_tokens[1:]
        P_T = TI_tokens[1:]

        # 计算所有方向的外来原型 f
        directions = [
            ('N', 'R'), ('T', 'R'),
            ('R', 'N'), ('T', 'N'),
            ('R', 'T'), ('N', 'T')
        ]
        f_dict = {}
        for src, tgt in directions:
            P_src = {'R': P_R, 'N': P_N, 'T': P_T}[src]
            P_tgt = {'R': P_R, 'N': P_N, 'T': P_T}[tgt]
            affinity = self.compute_affinity(P_tgt, P_src)  # target(作为query) <-> source
            F_src_to_tgt = self.aggregate_cross_features(P_tgt, P_src, affinity)
            f_dict[f'{src}→{tgt}'] = self.proto_pooling(F_src_to_tgt, f'{src}_to_{tgt}')

        # 计算 g 和 h
        _, _, _, h = self.compute_g_and_h(f_dict, c_R, c_N, c_T)

        # 原型进化
        P_new = self.evolve_proto(P_prev, h)

        # 原型反馈
        if self.cfg.MODEL.CPE_FEEDBACK_PROTO:
            gamma = self.gamma
            c_R_new = c_R + gamma * P_new
            c_N_new = c_N + gamma * P_new
            c_T_new = c_T + gamma * P_new
            # 将更新后的 CLS 写回 tokens
            RGB_tokens[0] = c_R_new.unsqueeze(0)
            NI_tokens[0] = c_N_new.unsqueeze(0)
            TI_tokens[0] = c_T_new.unsqueeze(0)

        return P_new

    def final_feature(self, all_protos, c_R, c_N, c_T):
        """
        根据收集的所有层原型和最终 CLS 生成最终检索特征
        all_protos: list of [B, D], length = num_layers
        """
        B = c_R.size(0)
        if self.layer_score is not None:
            # 可学习权重聚合多层原型
            scores = F.softmax(self.layer_score, dim=0)  # [L]
            P_final = torch.stack(all_protos, dim=0) * scores[:, None, None]  # [L, B, D]
            P_final = P_final.sum(dim=0)  # [B, D]
        else:
            P_final = all_protos[-1]  # 只用最后一层

        feat_cat = torch.cat([c_R, c_N, c_T, P_final], dim=-1)  # [B, 4D]
        final_feat = self.feat_proj(feat_cat)  # [B, 3D]
        return final_feat


# ========================= Transformer =========================
class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, cfg=None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.heads = heads
        self.cfg = cfg

        # 图像预处理
        self.conv1 = nn.Conv2d(3, width, kernel_size=16, stride=16, bias=False)
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(16 * 8 + 1, width))
        self.ln_pre = LayerNorm(width)

        # 共享 TCA 模块（HMA 使用）
        self.trainable_tca = nn.Sequential(
            nn.Linear(self.width, int(self.width * 2)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(self.width * 2), self.width),
        )

        # HMA Transformer Blocks
        self.resblocks = nn.Sequential(*[
            ResidualAttentionBlockWithHMA(width, heads, attn_mask, index=i, shared=self.trainable_tca)
            for i in range(layers)
        ])

        # 早期融合占位
        self.trainable_early_fusion = nn.Sequential(*[
            Nothing(self.width) for _ in range(layers)
        ])

        # 初始化 CPE 模块（仅在配置启用时）
        if cfg is not None and hasattr(cfg.MODEL, 'CPE_ENABLE') and cfg.MODEL.CPE_ENABLE:
            self.cpe = CrossModalPrototypeEvolution(d_model=width, layers=layers, cfg=cfg)
        else:
            self.cpe = None

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
        x = self.conv1(x)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)
        class_emb = self.class_embedding.to(x.dtype).to(x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        x = torch.cat([class_emb, x], dim=1)
        if cv_emb is not None:
            x[:, 0] = x[:, 0] + cv_emb.squeeze(1)
        x = x + self.positional_embedding.to(x.dtype).to(x.device)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # [L, B, D]
        return x

    def forward(self, RGB: torch.Tensor, NI: torch.Tensor, TI: torch.Tensor, cv_embed=None, modality=None):
        RGB = self.pre(RGB, cv_embed)
        NI = self.pre(NI, cv_embed)
        TI = self.pre(TI, cv_embed)

        all_protos = []               # 收集各层原型（未启用层记录 None）
        B = RGB.shape[1]
        P = torch.zeros(B, self.width, device=RGB.device, dtype=RGB.dtype)

        cpe_start = getattr(self.cfg.MODEL, 'CPE_START_LAYER', 0) if self.cfg is not None else 0

        for i in range(self.layers):
            RGB, NI, TI = self.trainable_early_fusion[i](RGB, NI, TI)

            if isinstance(self.resblocks[i], ResidualAttentionBlockWithHMA):
                RGB = self.resblocks[i](RGB, modality='rgb', index=i)
                NI = self.resblocks[i](NI, modality='nir', index=i)
                TI = self.resblocks[i](TI, modality='tir', index=i)
            else:
                RGB = self.resblocks[i](RGB)
                NI = self.resblocks[i](NI)
                TI = self.resblocks[i](TI)

            if self.cpe is not None and i >= cpe_start:
                c_R = RGB[0]
                c_N = NI[0]
                c_T = TI[0]
                P = self.cpe(i, P, RGB, NI, TI, c_R, c_N, c_T)
                all_protos.append(P)
            else:
                all_protos.append(None)   # 占位

        RGB = RGB.permute(1, 0, 2)
        NI = NI.permute(1, 0, 2)
        TI = TI.permute(1, 0, 2)

        if self.cpe is not None:
            return RGB, NI, TI, all_protos
        else:
            return RGB, NI, TI            # 关键修改


class Nothing(nn.Module):
    def __init__(self, width):
        super().__init__()
    def forward(self, RGB, NIR, TIR):
        return RGB, NIR, TIR