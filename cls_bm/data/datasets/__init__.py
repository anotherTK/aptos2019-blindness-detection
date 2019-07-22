
from .blindness import BlindnessDataset


_DATASETS = {
    "blindness": BlindnessDataset
}

def get(name):

    return _DATASETS[name]
    