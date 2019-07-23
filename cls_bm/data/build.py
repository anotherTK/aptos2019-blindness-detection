

import torch
from . import datasets as D
from . import samplers
from .transforms import build_transforms

from cls_bm.utils.comm import get_world_size

def build_dataset(dataset_name, root, stage, transforms):
    
    args = {
        "root": root,
        "stage": stage,
        "transforms": transforms
    }

    dataset = D.get(dataset_name)
    
    return dataset(**args)


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler
    

def make_batch_data_sampler(
    dataset, sampler, images_per_batch, num_iters=None, start_iter=0
):
    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, stage='train', is_distributed=False, start_iter=0, test_aug=False):
    num_gpus = get_world_size()
    if stage == 'train':
        images_per_gpu = cfg.SOLVER.IMS_PER_GPU
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        images_per_gpu = cfg.TEST.IMS_PER_GPU
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0
    
    if test_aug:
        transforms = build_transforms(cfg, True)
    else:
        transforms = build_transforms(cfg, stage=='train')
    dataset = build_dataset(cfg.DATA.DATASET, cfg.DATA.DATASET_ROOT, stage, transforms)

    sampler = make_data_sampler(dataset, shuffle, is_distributed)
    batch_sampler = make_batch_data_sampler(
        dataset, sampler, images_per_gpu, num_iters, start_iter
    )

    num_workers = cfg.DATA.NUM_WORKERS
    data_loader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_sampler=batch_sampler,
    )
    
    return data_loader
