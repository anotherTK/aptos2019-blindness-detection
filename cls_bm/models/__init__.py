
import functools
from .EfficientNet import EfficientNet


_MODELS = {
    "efficientnet-b0": functools.partial(EfficientNet.from_pretrained, "efficientnet-b0"),
    "efficientnet-b1": functools.partial(EfficientNet.from_pretrained, "efficientnet-b1"),
    "efficientnet-b2": functools.partial(EfficientNet.from_pretrained, "efficientnet-b2"),
    "efficientnet-b3": functools.partial(EfficientNet.from_pretrained, "efficientnet-b3"),
    "efficientnet-b4": functools.partial(EfficientNet.from_pretrained, "efficientnet-b4"),
    "efficientnet-b5": functools.partial(EfficientNet.from_pretrained, "efficientnet-b5"),
    "efficientnet-b6": functools.partial(EfficientNet.from_name, "efficientnet-b6"),
    "efficientnet-b7": functools.partial(EfficientNet.from_name, "efficientnet-b7"),
}

def build_model(cfg):

    model = _MODELS[cfg.MODEL.NAME](override_params={"num_classes": cfg.MODEL.NUM_CLASSES})
    
    return model

    
