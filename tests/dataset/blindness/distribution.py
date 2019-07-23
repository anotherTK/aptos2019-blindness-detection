
"""This explore the data distribution
"""

import os
import json
import argparse
from collections import Counter
import matplotlib.pyplot as plt

_CATS = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative DR"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Explore data distribution")
    parser.add_argument('--root', default='./datasets/blindness', help='Dataset root')
    parser.add_argument('--w', default='./work_dirs/blindness', help='Workspace')

    args = parser.parse_args()
    if not os.path.exists(args.w):
        os.makedirs(args.w)
    return args


def category_dst(labeled_data, save_dir):

    cats = []
    for itm in labeled_data:
        cats.append(itm[1])

    total_num = len(cats)
    c = Counter(cats)
    cat_keys = sorted(list(c.keys()))
    percentages = []
    
    for key in cat_keys:
        value = float(c[key]) / total_num * 100
        percentages.append(value)
        print("The cat [{}] occupies {}% in the labeled dataset.".format(key, value))

    cat_keys = [_CATS[e] for e in cat_keys]
    plt.bar(cat_keys, percentages, color='rgb')
    plt.savefig(os.path.join(save_dir, "cat_dst.png"))
    

def main(args):

    labeled_data = []
    train_data = json.load(open(os.path.join(args.root, 'train.json')))
    val_data = json.load(open(os.path.join(args.root, 'val.json')))

    labeled_data.extend(train_data)
    labeled_data.extend(val_data)

    category_dst(labeled_data, args.w)


if __name__ == "__main__":
    args = parse_args()
    main(args)
