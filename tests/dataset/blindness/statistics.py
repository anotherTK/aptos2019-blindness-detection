
import os
import cv2
import json
import argparse
import random
import numpy as np
from tqdm import tqdm
import multiprocessing as mp


def parse_args():
    parser = argparse.ArgumentParser(description="Calculate mean and std")
    parser.add_argument('--root', default='./datasets/blindness')
    parser.add_argument('--save', default='./work_dirs/blindness')
    parser.add_argument('--size', default=224, type=int, help="Image input size")
    parser.add_argument('--num', default=-1, type=int, help="The selected sample number, -1 means the whole dataset")
    parser.add_argument('--core', default=20, type=int, help="process number")

    args = parser.parse_args()

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    if args.core > mp.cpu_count():
        args.core = mp.cpu_count()
    return args


def do_job(images, root, size):
    
    h, w = size, size
    imgs = np.zeros([h, w, 3, 1])
    means, stds = [], []

    for image in images:
        filepath = os.path.join(root, image[0] + '.jpg')
        img = cv2.imread(filepath)
        img = cv2.resize(img, (w, h))
        img = img[:, :, :, np.newaxis]
        imgs = np.concatenate((imgs, img), axis=3)

    imgs = imgs.astype(np.float32) / 255.0
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        means.append(np.mean(pixels))
        stds.append(np.std(pixels))

    return means, stds

def main(args):

    images = []
    train_data = json.load(open(os.path.join(args.root, 'train.json')))
    val_data = json.load(open(os.path.join(args.root, 'val.json')))
    test_data = json.load(open(os.path.join(args.root, 'test.json')))

    images.extend(train_data)
    images.extend(val_data)
    images.extend(test_data)

    random.shuffle(images)

    selected_num = int(args.num)
    if selected_num < 0:
        selected_num = len(images)

    cores = args.core
    pool = mp.Pool(processes=cores)
    num_per_core = selected_num // cores + 1
    results = []
    for i in range(cores):
        results.append(pool.apply_async(
            do_job, (images[num_per_core * i: num_per_core * (i + 1)], args.root, args.size)))

    pool.close()
    pool.join()

    means = []
    stds = []

    for result in results:
        _mean, _std = result.get()
        means.append(_mean)
        stds.append(_std)

    means = np.array(means)
    stds = np.array(stds)
    means = np.mean(means, axis=0)
    stds = np.mean(stds, axis=0)

    means = means.tolist()
    stds = stds.tolist()

    # tranfer from BGR to RGB
    means.reverse()
    stds.reverse()

    with open(os.path.join(args.save, "statistics.txt"), 'w') as w_obj:
        w_obj.write("RGB mean: {}\n".format(means))
        w_obj.write("RGB std : {}\n".format(stds))

    



if __name__ == "__main__":
    args = parse_args()
    main(args)
