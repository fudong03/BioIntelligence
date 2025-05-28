import argparse
import os
import random
from collections import Counter

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-dr', '--data_root', type=str, default='/mnt/sdc/data/protein')
parser.add_argument('-lp', '--label_path', type=str, default='protein_details.csv')
args = parser.parse_args()

# number of proteins in each group
# Protease: 3067, Kinase: 2502, Receptor: 1437, Carbonic Anhydrase: 1206, Phosphatase: 431, Isomerase: 371
protein_labels = {'Protease': 0, 'Kinase': 1, 'Receptor': 2, 'Carbonic Anhydrase': 3, 'Phosphatase': 4, 'Isomerase': 5}


def write_config():
    # random shuffle
    seed = 3
    # train_val_ratio = 0.8
    train_val_ratio = 1.0

    detail_path = os.path.join(args.data_root, args.label_path)
    df = pd.read_csv(detail_path)

    lines = []
    for index, row in df.iterrows():
        line = '{} {} {}'.format(row['protein'], row['chain'], protein_labels[row['group']])
        lines.append(line)

    random.Random(seed).shuffle(lines)
    train_len = int(len(lines) * train_val_ratio)

    train_lines = lines[:train_len]
    test_lines = lines[train_len:]

    print("train: {}, test: {}".format(len(train_lines), len(test_lines)))

    train_labels = []
    for line in train_lines:
        _, _, label = line.split()
        train_labels.append(label)

    val_labels = []
    for line in test_lines:
        _, _, label = line.split()
        val_labels.append(label)

    train_label_count = Counter(train_labels)
    val_label_count = Counter(val_labels)
    print("train label counts: {} \nval label test: {}".format(train_label_count, val_label_count))

    train_path = '../data/protein_train.txt'
    test_path = '../data/protein_val.txt'

    with open(train_path, 'w') as f:
        f.write('\n'.join(train_lines))

    with open(test_path, 'w') as f:
        f.write('\n'.join(test_lines))


if __name__ == '__main__':
    write_config()
