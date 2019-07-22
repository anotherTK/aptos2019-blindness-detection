# APTOS2019-BLINDNESS-DETECTION

This repo solves the problem located in this [competition](https://www.kaggle.com/c/aptos2019-blindness-detection), from kaggle.

`Imagine being able to detect blindness before it happened.`

We using the images from aptos2019 and aptos2015, downloaded from kaggle data kernel.

# Usage

- extract the zip file into this data structure

```
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
```

- make soft link to datasets/blindness

- using the provided code, prepare train/val/test data

```sh
python tests/dataset/blindness/prepare_dataset.py
```