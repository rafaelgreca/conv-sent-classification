import numpy as np
import gensim
import torch
import torch.nn as nn
import os
import pandas as pd

from preprocessing import Preprocesser
from typing import Tuple
from collections.abc import Callable
from torch.utils.data import Dataset, DataLoader
from models import CNN

# will be used to create the data loader
class DatasetDL(Dataset):
    def __init__(self, tweets: list, labels: list) -> None:
        self.labels = labels
        self.tweets = tweets
    
    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        return self.tweets[idx], self.labels[idx]

# read the input files
def read_files(file_path: str,
               label: int):
    sentences = list()
    prep = Preprocesser()

    with open(file_path, errors="ignore") as f:
        for line in f:
            sentence = line.strip()
            cleaned_sentence = prep(str(sentence))
            sentences.append(cleaned_sentence)      

    df = pd.DataFrame({'sentence': sentences,
                       'label': [label] * len(sentences)})
    return df

# creates the vocabulary dict
# will be used to tokenize the data and to create the embedding
def create_vocab(sentences: list) -> dict:
    word_list = " ".join(sentences).split()
    word_list = list(set(word_list))
    word_list.sort() # making sure that we create the vocab in the same order every time we run
    word_dict = {w: i+1 for i, w in enumerate(word_list)}
    return word_dict

# prepares the data
# (tokenize, truncate and pad)
def prepare_data(input: str,
                 vocab: dict,
                 max_len: int) -> np.array:
    if len(input.split()) >= max_len:
        input = " ".join(input.split()[:max_len])
        tokenized_inputs = np.array([vocab[str(word)] for word in input.split()])
    else:
        input = [vocab[str(word)] for word in input.split()]
        input += [0] * (max_len - len(input))
        tokenized_inputs = np.array(input)
    return tokenized_inputs

# creates the data loader
def create_data_loader(data: list,
                       target: list,
                       batch_size: int,
                       worker_init: Callable,
                       generator: torch.Generator) -> DataLoader:
    pytorch_dataset = DatasetDL(data, target)
    pytorch_dataloader = DataLoader(pytorch_dataset,
                                    batch_size=batch_size,
                                    num_workers=0,
                                    worker_init_fn=worker_init,
                                    generator=generator,
                                    shuffle=False)
    return pytorch_dataloader

# loads a pre trained embedding
# e.g: glove/word2vec
def load_pretrained_embedding(path: str,
                              vocab: dict,
                              embedding_dim: int) -> np.array:
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))
    
    for word in vocab.keys():
        try:
            embedding_vector = model[word]
            if embedding_vector is not None:
                embedding_matrix[vocab[word]] = embedding_vector
            else:
                embedding_matrix[vocab[word]] = np.random.randn(embedding_dim)
        except KeyError:
            embedding_matrix[vocab[word]] = np.random.randn(embedding_dim)
    
    return embedding_matrix

# trains the model
def train_model(model: CNN,
                epochs: int,
                train_dataloader: DataLoader,
                validation_dataloader: DataLoader,
                device: torch.device) -> CNN:
    loss = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_loss = np.inf

    for epoch in range(epochs):
        for (idx, batch) in enumerate(train_dataloader):
            # training step
            model.train(True) # make sure gradient tracking is on, and do a pass over the data
            inputs = batch[0].to(device)
            targets = batch[1].type(torch.FloatTensor).to(device)
            outputs = model(inputs)

            l = loss(outputs, targets)
            l.backward()
            optimizer.step()
            optimizer.zero_grad()

            # validation step
            model.train(False)
            model.eval()
            running_vloss = 0.0

            for (v_idx, v_batch) in enumerate(validation_dataloader):
                v_inputs = v_batch[0].to(device)
                v_targets = v_batch[1].type(torch.FloatTensor).to(device)
                v_outputs = model(v_inputs)
                v_loss = loss(v_outputs, v_targets)
                running_vloss += v_loss.item()
            
            avg_vloss = running_vloss/(v_idx+1)

            if avg_vloss < best_loss:
                model_path = os.path.join(os.getcwd(), "model_checkpoints")
                os.makedirs(model_path, exist_ok=True)
                torch.save(model.state_dict(), f"{model_path}/{model.name}_{epoch}")
                best_loss = avg_vloss

            if idx==0:
                print(f"Epoch: {epoch+1} | Training loss: {l.item():1.6f} | Validation loss: {avg_vloss:1.6f}")

    return model