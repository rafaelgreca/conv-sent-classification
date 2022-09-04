import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import Union

class CNN(nn.Module):
    """
    Yoon Kim"s Convolutional Neural Network (CNN) models implementation described
    in https://arxiv.org/pdf/1408.5882.pdf. Models implemented so far: embedding randomly initialized
    (rand), embedding initialized with pretrained embedding's weights and not trainable (static) and 
    embedding initialized with pretrained embedding's weights and trainable (non-static).
    """
    def __init__(self,
                 model: str,
                 vocab_size: int,
                 embedding_dim: int,
                 embedding_weights: Union[None, np.array],
                 n_filters: int,
                 filter_sizes: list,
                 output_dim: int, 
                 dropout: float) -> None:
        """
        :param model: which model will be created ("rand", "static" or "non-static").
        :param vocab_size: the vocabulary length.
        :param embedding dim: the embedding dimmension.
        :param embedding_weights: the pretrained embedding weights (for the "static" and "non-static" model).
        :param n_filters: the number of filters in each convolutional layer.
        :param filter_sizes: the kernel/filter size in each convolutional layer.
        :param output_dim: the output dimmension (number of units in the dense layer).
        :param dropout: the droupout rate.
        """
        super(CNN, self).__init__()

        if model == 'rand':
            # create the embedding layer
            # randomly initialization
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
        else:
            # create the embedding layer
            # loading pretrained embedding
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)
            self.embedding.load_state_dict({"weight": embedding_weights})
            self.embedding.weight.requires_grad = False if model == 'static' else True

        # create the convolutional layers stack
        self.convs = nn.ModuleList([nn.Conv2d(1, n_filters, (fs, embedding_dim)) for fs in filter_sizes])

        # create the dropout layer
        self.dropout = nn.Dropout(dropout)

        # create the dense/fully connected layer (output layer)
        self.fc = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        
    def forward(self,
                text: torch.tensor) -> torch.tensor:
        # Embedding Layer
        embedded = self.embedding(text) # (batch size, sentence_len, embedding_dim)

        embedded = embedded.unsqueeze(1) # (batch size, 1, sentence_len, embedding_dim)

        # Convolutional layers
        convs_output = [F.relu(c(embedded)).squeeze(3) for c in self.convs]

        # Max pooling layers
        max_pools_output = [F.max_pool1d(o, o.size(2)).squeeze(2) for o in convs_output]

        # Concatanating max pooling layers outputs
        cat = self.dropout(torch.cat(max_pools_output, dim = 1))

        # Dense Layer
        output = torch.sigmoid(self.fc(cat)).squeeze(1)
        
        return output