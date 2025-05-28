# Do Protein Transformers Have Biological Intelligence?

## Model Overview

![SeqViT](input/method-seq-vit-arch.png)



Our proposed Sequence Protein Transformers (SPT) model consists of a stack of Transformer Encoders, and each of which comprises a Multi-head Self-Attention (MSA) block and an MLP block. Both of them incorporate a Layer Normalization before the block and a residual connection after the block. Different from prior Transformer variants the input sequence for our SPT model (i.e., a chain of amino acid embeddings)  is obtained by linearly projecting the one-hot embedding of the protein's primary structure.

Three model variants, i.e., SPT-Tiny, SPT-Small, and SPT-Base, are developed, tailored for protein function predictions across different scales of data. Specifically, all SPT variants are composed of 12 layers of Transformer blocks, with their hidden sizes set to 192, 384, and 768, and their number of heads set to 4, 6, and 12, respectively for the SPT-Tiny, the SPT-Small, and the SPT-Base models. The MLP sizes are consistently set to be four times of their corresponding hidden sizes.



## Dataset

- The `ProFunc-9K` dataset is available at [Google Drive](https://drive.google.com/drive/folders/1IXdK075Sw88k_Pj_GqdIpMJB7Q9rs2UU?usp=sharing)



## Requirements

- torch == 1.13.0
- torchvision == 0.14.0
- numpy == 1.26.0
- pandas == 2.0.3
- argparse == 1.4.0
- timm == 0.5.4
- scipy == 1.11.3
- biopandas == 0.4.1
- grad-cam == 1.4.8
- tensorboard == 2.13.0
- einops == 0.7.0

You can use the following instructions to install all the requirements:
```
pip install -r requirements.txt
```



## Experiments

- **The ProFunc-9K Dataset**

To reproduce our experiments, please download our ProFunc-9K dataset via the above Google Drive, and then move the dataset to a directory, e.g., `/mnt/data/protein`. Now, you can train our SPT model via the following command:

```shell
# train the SPT-Tiny model
python main_finetune_spt.py --data_root /mnt/data/protein/data --model spt_tiny_embed192
```



- **The Antibiotic Resistance (AR) and the Metal Ion Binding (MIB) Datasets**

To reproduce our experiments on the AR and MIB datasets, please use the following command:

```shell
# train the SPT-Tiny model on the AR dataset
python main_finetune_spt_benchmarks.py --data_root ./data/Antibiotic\ Resistance --model spt_tiny_embed192 --max_seq 1024 --nb_classes 19
```



## License

This repository is under the CC-BY-NC 4.0 license. Please refer to the `LICENSE`  file for details.
