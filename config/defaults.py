from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# MODEL
# -----------------------------------------------------------------------------
_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.DEVICE_ID = '0'
_C.MODEL.NAME = 'HAPE'
_C.MODEL.PRETRAIN_PATH_T = './pretrain/ViT-B-16.pt'

# 基础训练配置
_C.MODEL.NECK = 'bn'               # BNNeck 类型
_C.MODEL.ID_LOSS_TYPE = 'softmax'
_C.MODEL.ID_LOSS_WEIGHT = 1.0
_C.MODEL.TRIPLET_LOSS_WEIGHT = 1.0
_C.MODEL.METRIC_LOSS_TYPE = 'triplet'   # 只使用 triplet loss
_C.MODEL.DIST_TRAIN = False
_C.MODEL.FROZEN = False            # 是否冻结 backbone
_C.MODEL.IF_LABELSMOOTH = 'on'     # 标签平滑

# Transformer 基础配置
_C.MODEL.TRANSFORMER_TYPE = 'ViT-B-16'
_C.MODEL.STRIDE_SIZE = [16, 16]

# ========== HMA 模块配置 ==========
_C.MODEL.HMA_ENABLE = False
_C.MODEL.FLOPS_TEST = False

# ========== CPE 模块配置 ==========
_C.MODEL.CPE_ENABLE = False
_C.MODEL.CPE_START_LAYER = 6
_C.MODEL.CPE_LEARNABLE_AFFINITY = True
_C.MODEL.CPE_USE_CROSS_ATTN = True
_C.MODEL.CPE_PROTO_POOLING = 'attn'
_C.MODEL.CPE_MULTI_MODAL_FUSION = 'transformer'
_C.MODEL.CPE_EVOLVE_TYPE = 'gru'
_C.MODEL.CPE_FEEDBACK_PROTO = True
_C.MODEL.CPE_LAYER_AGGR = True
_C.MODEL.CPE_RETRIEVAL_FEAT = 'mlp'

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
_C.INPUT.SIZE_TRAIN = [256, 128]
_C.INPUT.SIZE_TEST = [256, 128]
_C.INPUT.PROB = 0.5
_C.INPUT.RE_PROB = 0.5
_C.INPUT.PIXEL_MEAN = [0.5, 0.5, 0.5]
_C.INPUT.PIXEL_STD = [0.5, 0.5, 0.5]
_C.INPUT.PADDING = 10

# -----------------------------------------------------------------------------
# DATASETS
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.NAMES = ('RGBNT201')
_C.DATASETS.ROOT_DIR = './data'

# -----------------------------------------------------------------------------
# DATALOADER
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 8
_C.DATALOADER.SAMPLER = 'softmax_triplet'
_C.DATALOADER.NUM_INSTANCE = 8

# -----------------------------------------------------------------------------
# SOLVER
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "Adam"
_C.SOLVER.MAX_EPOCHS = 50
_C.SOLVER.BASE_LR = 0.00035
_C.SOLVER.MARGIN = 0.3
_C.SOLVER.WEIGHT_DECAY = 0.0001
_C.SOLVER.WEIGHT_DECAY_BIAS = 0.0001
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (40, 70)
_C.SOLVER.WARMUP_FACTOR = 0.01
_C.SOLVER.WARMUP_ITERS = 10
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.SEED = 1111
_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 10
_C.SOLVER.EVAL_PERIOD = 1
_C.SOLVER.IMS_PER_BATCH = 64
_C.SOLVER.HMA_LR_FACTOR = 1.0

# -----------------------------------------------------------------------------
# TEST
# -----------------------------------------------------------------------------
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 128
_C.TEST.RE_RANKING = 'no'
_C.TEST.WEIGHT = ""
_C.TEST.NECK_FEAT = 'before'
_C.TEST.FEAT_NORM = 'yes'

# -----------------------------------------------------------------------------
# OUTPUT
# -----------------------------------------------------------------------------
_C.OUTPUT_DIR = "./output"
_C.MODEL.NO_MARGIN = True