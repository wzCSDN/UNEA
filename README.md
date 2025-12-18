# UNEA: Unsupervised Entity Alignment Based on Personalized Discriminative Rooted Tree

A PyTorch implementation for UNEA.

## Quick Start

Follow these three steps to run the model.

### 1. Download Datasets

First, run the script to download the DBP15K dataset and the pre-trained LaBSE model.

```bash
cd data
bash getdata.sh
cd ..
```

### 2. Generate Initial Embeddings
Next, generate initial entity embeddings using the pre-trained LaBSE model.

```bash
python script/preprocess/labse_dump.py
```
### 3. Run Training
Finally, start the training process.

```bash
python run.py
```
