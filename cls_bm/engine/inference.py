
import logging
import time
import os

import torch
from tqdm import tqdm
import numpy as np

from cls_bm.config import cfg
from cls_bm.data import make_data_loader
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str

def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            outputs = model(images.to(device))
            if timer:
                if not cfg.MODEL.DEVICE == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            # top-1 accuracy
            _, preds = outputs.topk(1, 1, True, True)
            
            if isinstance(targets, torch.Tensor):
                preds = preds.t()
                targets = targets.to(device)
                batch_size = targets.size(0)
                correct = preds.eq(targets.view(1, -1).expand_as(preds))
                correct_num = correct[:1].view(-1).float().sum(0, keepdim=True)
                if "correct" not in results_dict:
                    results_dict["correct"] = 0
                    results_dict["total"] = 0
                results_dict["correct"] += correct_num.to(cpu_device)
                results_dict["total"] += batch_size
            else:
                preds = preds.squeeze(-1)
                preds = [o.to(cpu_device).item() for o in preds]
                results_dict.update(
                    {img_id: result for img_id, result in zip(targets, preds)}
                )
    return results_dict


def _accumulate_predictions_from_multiple_gpus(predictions_per_gpu):
    all_predictions = all_gather(predictions_per_gpu)
    if not is_main_process():
        return
    # merge the list of dicts
    predictions = {}
    for p in all_predictions:
        if "correct" in p and "total" in p and len(p.keys()) == 2:
            if "correct" not in predictions:
                predictions["correct"] = 0
                predictions["total"] = 0
            predictions["correct"] += p["correct"]
            predictions["total"] += p["total"]
        else:
            predictions.update(p)
    
    return predictions


def inference(
        model,
        data_loader,
        dataset_name,
        device="cuda",
        output_folder=None,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("cls_bm.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(
        dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(
        model, data_loader, device, inference_timer)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time *
            num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    predictions = _accumulate_predictions_from_multiple_gpus(predictions)
    if not is_main_process():
        return

    if output_folder:
        torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    if len(predictions.keys()) == 2:
        print(predictions)


def inference_aug(
    cfg,
    model,
    dataset_name,
    device="cuda",
    output_folder=None,
    aug_times=10,
    distributed=False,
):
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("cls_bm.inference")
    
    total_predictions = {}
    for i in range(aug_times):
        data_loader = make_data_loader(
            cfg, stage='test', is_distributed=distributed, test_aug=True)
        dataset = data_loader.dataset
        logger.info("Start evaluation on {} dataset({} images).".format(
            dataset_name, len(dataset)))
        total_timer = Timer()
        inference_timer = Timer()
        total_timer.tic()
        predictions = compute_on_dataset(
            model, data_loader, device, inference_timer)
        # wait for all processes to complete before measuring the time
        synchronize()
        total_time = total_timer.toc()
        total_time_str = get_time_str(total_time)
        logger.info(
            "Total run time: {} ({} s / img per device, on {} devices)".format(
                total_time_str, total_time *
                num_devices / len(dataset), num_devices
            )
        )
        total_infer_time = get_time_str(inference_timer.total_time)
        logger.info(
            "Model inference time: {} ({} s / img per device, on {} devices)".format(
                total_infer_time,
                inference_timer.total_time * num_devices / len(dataset),
                num_devices,
            )
        )

        predictions = _accumulate_predictions_from_multiple_gpus(predictions)
        if is_main_process():
            for img_id, result in predictions.items():
                if img_id not in total_predictions:
                    total_predictions[img_id] = []
                total_predictions[img_id].append(result)
        
    if not is_main_process():
        return

    # filter
    predictions_aug = {}
    for img_id, results in total_predictions.items():
        bins = np.bincount(results)
        pred = np.argmax(bins)
        predictions_aug[img_id] = pred
        
    if output_folder:
        torch.save(predictions_aug, os.path.join(
            output_folder, "predictions_aug.pth"))
