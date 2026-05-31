import argparse
import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch import nn
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import cfg
from data.datasets.make_dataloader import make_dataloader
from modeling import make_model
from utils.logger import setup_logger


# ---------- 自定义输入字典 ----------
class Newdict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape = [1, 2, 3, 4]

    def to(self, device):
        for key in self.keys():
            self[key] = self[key].to(device)
        return self

    def size(self, k):
        data = self['RGB']
        w, h = data.size(-1), data.size(-2)
        return w if k == -1 else h


# ---------- 单模态模型包装器 ----------
class SingleModalityWrapper(nn.Module):
    def __init__(self, full_model, modality):
        super().__init__()
        self.full_model = full_model
        self.modality = modality
        self.backbone = full_model.BACKBONE

    def forward(self, x_dict):
        RGB = x_dict['RGB']
        NI = x_dict['NI']
        TI = x_dict['TI']
        cam_label = x_dict.get('cam_label', 0)
        view_label = x_dict.get('view_label', None)

        out = self.backbone(RGB, NI, TI, cam_label=cam_label, view_label=view_label)

        if self.full_model.cpe_enable:
            _, RGB_global, _, NI_global, _, TI_global, _ = out
        else:
            _, RGB_global, _, NI_global, _, TI_global = out

        feat = {'RGB': RGB_global, 'NI': NI_global, 'TI': TI_global}[self.modality]
        feat_bn = self.backbone.bottleneck(feat)
        score = self.backbone.classifier(feat_bn)
        return score


# ---------- 可视化函数 ----------
def show_cam(index, imgpath, grayscale_cam, modality, model_name, cfg, n_iter):
    index = int(index)
    img_name = imgpath[index]
    print(f"Processing {model_name} {modality}: {img_name}")

    if cfg.DATASETS.NAMES == 'RGBNT201':
        img_path = f'../RGBNT201/test/{modality}/{img_name}'
    elif cfg.DATASETS.NAMES == 'RGBNT100':
        img_path = f'../RGBNT100/rgbir/query/{img_name}'
    else:
        raise ValueError(f"Unsupported dataset: {cfg.DATASETS.NAMES}")

    grayscale_cam = grayscale_cam[index]

    if cfg.DATASETS.NAMES == 'RGBNT100':
        img = Image.open(img_path).convert('RGB')
        if modality == "RGB":
            cropped_image = img.crop((0, 0, 256, 128))
        elif modality == "NI":
            cropped_image = img.crop((256, 0, 512, 128))
        else:
            cropped_image = img.crop((512, 0, 768, 128))
        rgb_img = np.float32(cropped_image) / 255
    else:
        img = cv2.imread(img_path, 1)
        img = cv2.resize(img, (128, 256))
        rgb_img = np.float32(img) / 255

    visualization = show_cam_on_image(rgb_img, grayscale_cam)

    output_dir = f'../gradcam_vis/{cfg.DATASETS.NAMES}/{model_name}/{modality}'
    os.makedirs(output_dir, exist_ok=True)
    save_path = f'{output_dir}/{n_iter * cfg.TEST.IMS_PER_BATCH + index}.jpg'
    cv2.imwrite(save_path, visualization)


# ---------- 主程序 ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HAPE Grad‑CAM Baseline vs Improvement")
    parser.add_argument("--config_file", default="", help="Path to config file", type=str)
    parser.add_argument("--baseline_weight", type=str, required=True, help="Path to baseline model weights (.pth)")
    parser.add_argument("--improved_weight", type=str, required=True, help="Path to improved model weights (.pth)")
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,nargs=argparse.REMAINDER)
    args = parser.parse_args()

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
    logger = setup_logger("HAPE", output_dir, if_train=True)
    logger.info(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    device = "cuda"

    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    # ---------- 构建基线模型（关闭 HMA 和 CPE） ----------
    logger.info("Building baseline model (HMA=False, CPE=False)")
    cfg_baseline = cfg.clone()
    cfg_baseline.defrost()
    cfg_baseline.MODEL.HMA_ENABLE = False
    cfg_baseline.MODEL.CPE_ENABLE = False
    cfg_baseline.freeze()
    model_baseline = make_model(cfg_baseline, num_class=num_classes, camera_num=camera_num, view_num=view_num)

    # 加载基线权重（只保留形状匹配的参数）
    state_dict_b = torch.load(args.baseline_weight, map_location='cpu')
    model_dict_b = model_baseline.state_dict()
    filtered_b = {}
    for k, v in state_dict_b.items():
        if k in model_dict_b and v.shape == model_dict_b[k].shape:
            filtered_b[k] = v
        else:
            print(f"[Baseline] Skipping {k}: {v.shape} vs {model_dict_b[k].shape if k in model_dict_b else 'missing'}")
    missing_b, unexpected_b = model_baseline.load_state_dict(filtered_b, strict=False)
    print(f"[Baseline] Loaded: {len(filtered_b)}, Missing keys: {len(missing_b)}, Unexpected keys: {len(unexpected_b)}")
    model_baseline.eval().to(device)

    # ---------- 构建改进模型 ----------
    logger.info("Building improved model (from config)")
    model_improved = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num=view_num)
    state_dict_i = torch.load(args.improved_weight, map_location='cpu')
    model_dict_i = model_improved.state_dict()
    filtered_i = {}
    for k, v in state_dict_i.items():
        if k in model_dict_i and v.shape == model_dict_i[k].shape:
            filtered_i[k] = v
        else:
            print(f"[Improved] Skipping {k}: {v.shape} vs {model_dict_i[k].shape if k in model_dict_i else 'missing'}")
    missing_i, unexpected_i = model_improved.load_state_dict(filtered_i, strict=False)
    print(f"[Improved] Loaded: {len(filtered_i)}, Missing keys: {len(missing_i)}, Unexpected keys: {len(unexpected_i)}")
    model_improved.eval().to(device)

    # ---------- 定义目标层 ----------
    target_layers_baseline = [model_baseline.BACKBONE.base.resblocks[-1].ln_2]
    target_layers_improved = [model_improved.BACKBONE.base.resblocks[-1].ln_2]

    def reshape_transform(tensor, height=16, width=8):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        return result.transpose(2, 3).transpose(1, 2)

    # ---------- 生成热力图 ----------
    for n_iter, (img, pid, camids, camids_batch, viewids, imgpath) in enumerate(val_loader):
        # 仅取第一张图片，降低显存压力
        single_img = {
            'RGB': img['RGB'][0:1].to(device),   # 保持 batch 维度 [1, C, H, W]
            'NI': img['NI'][0:1].to(device),
            'TI': img['TI'][0:1].to(device)
        }
        single_camids = camids[0:1].to(device)
        single_imgpath = [imgpath[0]]   # 只有一张图

        x = Newdict({
            'RGB': single_img['RGB'],
            'NI': single_img['NI'],
            'TI': single_img['TI'],
            'cam_label': single_camids
        })

        modalities = ['RGB', 'NI', 'TI']

        with torch.no_grad():
            # 基线模型
            for mod in modalities:
                wrapper = SingleModalityWrapper(model_baseline, mod)
                cam = GradCAM(model=wrapper, target_layers=target_layers_baseline,
                            reshape_transform=reshape_transform)
                grayscale_cam = cam(input_tensor=x)
                show_cam(0, single_imgpath, grayscale_cam, mod, "baseline", cfg, n_iter)
                del cam, grayscale_cam, wrapper
                torch.cuda.empty_cache()

            # 改进模型
            for mod in modalities:
                wrapper = SingleModalityWrapper(model_improved, mod)
                cam = GradCAM(model=wrapper, target_layers=target_layers_improved,
                            reshape_transform=reshape_transform)
                grayscale_cam = cam(input_tensor=x)
                show_cam(0, single_imgpath, grayscale_cam, mod, "improved", cfg, n_iter)
                del cam, grayscale_cam, wrapper
                torch.cuda.empty_cache()

        # 处理一张图就退出，若需要多张可去掉 break 并循环 single_img 索引
        break