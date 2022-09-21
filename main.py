import argparse
import sys
import torch
import random
import os
import numpy as np
import warnings
import pandas as pd
import time
import torch.nn as nn
from utils import create_vocab, prepare_data, load_pretrained_embedding, read_files
from sklearn.model_selection import train_test_split
from models import CNN, SaveBestModel
from torch.utils.data import DataLoader
from dataset import DatasetDL
from torchtext.datasets import SST
from torchtext.data import Field, BucketIterator

warnings.filterwarnings("ignore")
seed = 2109
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(1)

# making sure the experiment is reproducible
os.environ["PYTHONHASHSEED"] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if you are using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def _init_fn(worker_id):
    np.random.seed(seed)
    random.seed(seed)

generator = torch.Generator()
generator.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Which model use (rand, static, non-static)", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size for training and evaluation", default=50)
    parser.add_argument("--embedding_path", type=str, help="Embedding file path", default="")
    parser.add_argument("--embedding_dim", type=int, help="Embedding dimmension", default=300)
    parser.add_argument("--epochs", type=int, help="Epochs in training step for the CNN model", default=10)
    parser.add_argument("--max_len", type=int, help="Sequence max length", default=30)
    parser.add_argument("--test_size", type=float, help="Test size (percentage)", default=0.1)
    parser.add_argument("--dropout_rate", type=float, help="Dropout rate", default=0.5)
    parser.add_argument("--filter_units", type=int, help="Filter units", default=100)
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        raise Exception("No arguments passed")
    else:
        available_models = ["rand", "static", "non-static"]
        
        assert args.model in available_models, f"You must provide a valid model. Valid models: {available_models}"
        assert (args.dropout_rate <= 1.0) and (args.dropout_rate > 0), "You must provide a valid dropout rate (between 0 and 1)"
        assert (args.test_size <= 1.0) and (args.test_size > 0), "You must provide a valid test size (between 0 and 1)"

        model_name = args.model
        text_field = Field(lower=True)
        label_field = Field(sequential=False)
        
        train_data, dev_data, test_data = SST.splits(text_field, label_field, fine_grained=True)

        text_field.build_vocab(train_data, dev_data, test_data)
        label_field.build_vocab(train_data, dev_data, test_data)
        
        train_iter, dev_iter, test_iter = BucketIterator.splits((train_data, dev_data, test_data), 
                                                                batch_sizes=(args.batch_size, 
                                                                            len(dev_data), 
                                                                            len(test_data)))