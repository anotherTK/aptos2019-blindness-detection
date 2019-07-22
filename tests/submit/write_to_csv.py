
import os
import torch
import argparse
import csv


def parse_args():
    
    parser = argparse.ArgumentParser(description="Translate classification results into csv style.")
    parser.add_argument('--c', help="classification results")
    parser.add_argument('--s', default='./work_dirs/submit_blindness.csv', help="File to be submitted.")

    args = parser.parse_args()
    return args


def main(args):

    predictions = torch.load(args.c)
    img_ids = predictions.keys()
    img_ids = sorted(img_ids)
    csv_head = ['id_code', 'diagnosis']
    
    with open(args.s, "w") as w_obj:
        csv_writer = csv.writer(w_obj)
        csv_writer.writerow(csv_head)
        
        for img_id in img_ids:
            csv_writer.writerow([img_id, predictions[img_id]])

    print("The results has saved to {}".format(args.s))


if __name__ == "__main__":
    
    args = parse_args()
    main(args)