"""data structure
├── images
│   ├── test_15
│   ├── test_19
│   ├── train_15
│   └── train_19
├── labels
│   ├── test_15.csv
│   ├── test_19.csv
│   ├── train_15.csv
│   └── train_19.csv
"""


import os
import json
import argparse
import random
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Argument data with aptos15 & aptos 19")
    parser.add_argument('--root', default='datasets/blindness')
    parser.add_argument('--save', default='datasets/blindness')
    
    args = parser.parse_args()
    return args


def main(args):

    train_data = []
    val_data = []
    test_data = []

    labels = os.listdir(os.path.join(args.root, 'labels'))
    for label in labels:
        image_folder = 'images/' + label.split('.')[0]
        csv_data = pd.read_csv(os.path.join(args.root, 'labels', label))
        for i in range(len(csv_data)):
            data = csv_data.loc[i]
            image_name = data.get('image', data.get('id_code'))
            image_cls = data.get('level', data.get('diagnosis'))
            if image_cls is None:
                test_data.append((os.path.join(image_folder, image_name), image_cls))
            else:
                train_data.append((os.path.join(image_folder, image_name), int(image_cls)))


    # split into train/val 19/1
    labeled_num = len(train_data)
    val_num = int(labeled_num * 0.05)
    random.shuffle(train_data)
    val_data = train_data[:val_num]
    train_data = train_data[val_num:]

    # write to json
    with open(os.path.join(args.save, 'train.json'), 'w') as w_obj:
        json.dump(train_data, w_obj)
    with open(os.path.join(args.save, 'val.json'), 'w') as w_obj:
        json.dump(val_data, w_obj)
    with open(os.path.join(args.save, 'test.json'), 'w') as w_obj:
        json.dump(test_data, w_obj)



if __name__ == "__main__":
    
    args = parse_args()
    main(args)
