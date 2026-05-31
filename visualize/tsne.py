import argparse
import os
import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from config import cfg
from data.datasets.make_dataloader import make_dataloader
from modeling import make_model
from utils.logger import setup_logger


def extract_features(model, dataloader, device, max_samples=None):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for img, pid, camids, camids_batch, viewids, imgpath in dataloader:
            # 处理可能嵌套的 pid
            if isinstance(pid, (tuple, list)):
                pid = pid[0]
            if not torch.is_tensor(pid):
                pid = torch.tensor(pid)
            pid = pid.view(-1)   # ← 关键：确保至少一维，避免标量

            x = {
                'RGB': img['RGB'].to(device),
                'NI': img['NI'].to(device),
                'TI': img['TI'].to(device),
                'cam_label': camids.to(device)
            }
            feat = model(x)       # 返回融合特征 [B, 3*feat_dim]
            features.append(feat.cpu())
            labels.append(pid)
            if max_samples and len(torch.cat(labels)) >= max_samples:
                break
    features = torch.cat(features, dim=0).numpy()
    labels = torch.cat(labels, dim=0).numpy()
    return features, labels


def plot_tsne(features_b, labels_b, features_i, labels_i, save_path, perplexity=30):
    # 合并所有特征
    all_features = np.concatenate([features_b, features_i], axis=0)

    # 先 PCA 降维到 50，减少 t‑SNE 的计算量并去除噪声
    from sklearn.decomposition import PCA
    pca = PCA(n_components=50, random_state=42)
    all_features = pca.fit_transform(all_features)

    # 固定线程数，避免 threadpoolctl 冲突
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, n_jobs=1)
    all_embeddings = tsne.fit_transform(all_features)

    n_b = features_b.shape[0]
    emb_b = all_embeddings[:n_b]
    emb_i = all_embeddings[n_b:]

    # 颜色映射
    all_ids = np.unique(np.concatenate([labels_b, labels_i]))
    id_to_color = {pid: i for i, pid in enumerate(all_ids)}
    colors_b = [id_to_color[pid] for pid in labels_b]
    colors_i = [id_to_color[pid] for pid in labels_i]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    scatter1 = axes[0].scatter(emb_b[:, 0], emb_b[:, 1], c=colors_b, cmap='tab20', s=5, alpha=0.7)
    axes[0].set_title('Baseline')
    axes[0].axis('off')

    scatter2 = axes[1].scatter(emb_i[:, 0], emb_i[:, 1], c=colors_i, cmap='tab20', s=5, alpha=0.7)
    axes[1].set_title('Improved')
    axes[1].axis('off')

    cbar = fig.colorbar(scatter2, ax=axes, orientation='horizontal', fraction=0.05, pad=0.08)
    cbar.set_label('Identity ID')

    plt.suptitle(f't‑SNE Feature Visualization (perplexity={perplexity})')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"t‑SNE plot saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="t‑SNE visualization for HAPE models")
    parser.add_argument("--config_file", default="", help="Path to config file", type=str)
    parser.add_argument("--baseline_weight", type=str, required=True, help="Path to baseline model weights (.pth)")
    parser.add_argument("--improved_weight", type=str, required=True, help="Path to improved model weights (.pth)")
    parser.add_argument("--perplexity", type=int, default=30, help="t‑SNE perplexity")
    parser.add_argument("--max_samples", type=int, default=None, help="Max number of samples to use (None for all)")
    parser.add_argument("--output", type=str, default="tsne_comparison.png", help="Output image path")
    parser.add_argument("opts", help="Modify config options via command line", default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()

    # ---------- 配置加载 ----------
    if args.config_file:
        import yaml
        from yacs.config import CfgNode
        with open(args.config_file, 'r', encoding='utf-8') as f:
            cfg_dict = yaml.safe_load(f)
        cfg_new = CfgNode(cfg_dict)
        cfg.merge_from_other_cfg(cfg_new)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)
    logger = setup_logger("HAPE", output_dir, if_train=False)
    logger.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载数据集 (仅需要验证集)
    _, _, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # ---------- 构建基线模型 ----------
    logger.info("Building baseline model (HMA=False, CPE=False)")
    cfg_baseline = cfg.clone()
    cfg_baseline.defrost()
    cfg_baseline.MODEL.HMA_ENABLE = False
    cfg_baseline.MODEL.CPE_ENABLE = False
    cfg_baseline.freeze()
    model_baseline = make_model(cfg_baseline, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # 加载权重（过滤形状不匹配的参数）
    state_dict_b = torch.load(args.baseline_weight, map_location='cpu')
    model_dict_b = model_baseline.state_dict()
    filtered_b = {}
    for k, v in state_dict_b.items():
        if k in model_dict_b and v.shape == model_dict_b[k].shape:
            filtered_b[k] = v
    model_baseline.load_state_dict(filtered_b, strict=False)
    model_baseline.to(device)
    model_baseline.eval()

    # ---------- 构建改进模型 ----------
    logger.info("Building improved model (from config)")
    model_improved = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    state_dict_i = torch.load(args.improved_weight, map_location='cpu')
    model_dict_i = model_improved.state_dict()
    filtered_i = {}
    for k, v in state_dict_i.items():
        if k in model_dict_i and v.shape == model_dict_i[k].shape:
            filtered_i[k] = v
    model_improved.load_state_dict(filtered_i, strict=False)
    model_improved.to(device)
    model_improved.eval()

    # ---------- 提取特征 ----------
    logger.info("Extracting baseline features...")
    feats_b, labels_b = extract_features(model_baseline, val_loader, device, args.max_samples)
    logger.info(f"Baseline: {len(feats_b)} samples")

    logger.info("Extracting improved features...")
    feats_i, labels_i = extract_features(model_improved, val_loader, device, args.max_samples)
    logger.info(f"Improved: {len(feats_i)} samples")

    # ---------- t‑SNE 并绘图 ----------
    plot_tsne(feats_b, labels_b, feats_i, labels_i, args.output, args.perplexity)