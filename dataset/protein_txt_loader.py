import torch
import os
from biopandas.pdb import PandasPdb
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# Amino acid alphabet and mapping with 1-letter codes
amino_acids = {
    'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4,
    'G': 5, 'H': 6, 'I': 7, 'K': 8, 'L': 9,
    'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
    'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19
}

class ProteinTxtDataset(Dataset):

    def __init__(self, root, txt):

        txt_path = os.path.join(root, txt)
        self.proteins = []

        self.labels = []

        with open(txt_path) as f:
            for line in f:
                protein, label = line.split()

                self.proteins.append(protein)
                self.labels.append(int(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):

        sequence = self.proteins[index]
        label = self.labels[index]

        embed = self.one_hot_embedding(sequence)
        # convert int to float
        embed = embed.astype(np.float32)

        return embed, label

    def one_hot_embedding(self, sequence):
        """
        Creates one-hot embeddings for a given amino acid sequence using numpy.

        Args:
            sequence (str): Amino acid sequence using 1-letter codes.

        Returns:
            numpy.ndarray: One-hot embeddings of the amino acid sequence.
        """
        num_amino_acids = len(amino_acids)
        embedding = np.zeros((len(sequence), num_amino_acids), dtype=int)
        for i, aa in enumerate(sequence):
            if aa in amino_acids:
                embedding[i, amino_acids[aa]] = 1
        return embedding


def save_list_to_csv(int_list, file_path):
    # Create a DataFrame from the list of integers
    df = pd.DataFrame({'Seq Length': int_list})

    # Save the DataFrame to a CSV file
    df.to_csv(file_path, index=True)


if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)

    config_root = "./../data/Metal Ion Binding"
    # config_root = "./../data/Antibiotic Resistance"
    # pdb_root = "/mnt/data/protein/data"

    val_set = ProteinTxtDataset(config_root,  "train.txt")
    # val_set = ProteinTxtDataset(config_root, "test.txt")

    # train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=8)

    max_seq_len = float('-inf')
    min_seq_len = float('inf')
    for i, (embed, label) in enumerate(val_loader):
        print(embed.shape, label.shape)

        max_seq_len = max(max_seq_len, embed.shape[1])
        min_seq_len = min(min_seq_len, embed.shape[1])


    print()
    print("max_seq_len: {}, min_seq_len: {}".format(max_seq_len, min_seq_len))

