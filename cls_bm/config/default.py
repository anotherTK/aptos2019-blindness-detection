
from yacs.config import CfgNode as CN

_C = CN()
_C.OUTPUT_DIR = "work_dirs/"
_C.DTYPE = "float32"

_C.MODEL = CN()
_C.MODEL.NAME = "efficientnet-b4"
_C.MODEL.NUM_CLASSES = 5
_C.MODEL.DEVICE = "cuda"
_C.MODEL.WEIGHT = ""

_C.DATA = CN()
_C.DATA.DATASET = "blindness"
_C.DATA.DATASET_ROOT = "datasets/blindness"
_C.DATA.NUM_WORKERS = 8
_C.DATA.HORIZON_FLIP_PROB_TRAIN = 0.5
_C.DATA.VERTICAL_FLIP_PROB_TRAIN = 0.5
_C.DATA.RORATION_PROB_TRAIN = 0.5
_C.DATA.ROTATION_DEGREES = 120
_C.DATA.BRIGHTNESS = 0.0
_C.DATA.CONTRAST = 0.0
_C.DATA.SATURATION = 0.0
_C.DATA.HUE = 0.0
_C.DATA.INPUT_SIZE = 380
_C.DATA.TO_BGR255 = True
_C.DATA.PIXEL_MEAN = [102.9801, 115.9465, 122.7717]
_C.DATA.PIXEL_STD = [1., 1., 1.]


_C.SOLVER = CN()
_C.SOLVER.IMS_PER_GPU = 2
_C.SOLVER.MAX_ITER = 40000
_C.SOLVER.BASE_LR = 0.01
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.BIAS_LR_FACTOR = 2
_C.SOLVER.WEIGHT_DECAY_BIAS = 0
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.STEPS = (30000, )
_C.SOLVER.GAMMA = 0.1
_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"
_C.SOLVER.CHECKPOINT_PERIOD = 2500


_C.TEST = CN()
_C.TEST.IMS_PER_GPU = 1
