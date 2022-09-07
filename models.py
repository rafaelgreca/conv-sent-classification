import os
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from typing import Union

# TODO: Refazer
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

        # create the convolutional layers
        self.conv1 = nn.Conv2d(1, n_filters, (filter_sizes[0], embedding_dim))
        self.conv2 = nn.Conv2d(1, n_filters, (filter_sizes[1], embedding_dim))
        self.conv3 = nn.Conv2d(1, n_filters, (filter_sizes[2], embedding_dim))
        
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
        output_conv1 = F.relu(self.conv1(embedded).squeeze(3))
        output_conv2 = F.relu(self.conv2(embedded).squeeze(3))
        output_conv3 = F.relu(self.conv3(embedded).squeeze(3))

        # Max pooling layers
        output_maxpool1 = F.max_pool1d(output_conv1, output_conv1.size(2)).squeeze(2)
        output_maxpool2 = F.max_pool1d(output_conv2, output_conv2.size(2)).squeeze(2)
        output_maxpool3 = F.max_pool1d(output_conv3, output_conv3.size(2)).squeeze(2)
        
        output_maxpool = torch.cat((output_maxpool1, output_maxpool2, output_maxpool3), dim = 1)
        
        # Dropout
        cat = self.dropout(output_maxpool)

        # Dense Layer
        output = self.fc(cat)
        
        return output
    
class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
        os.makedirs(os.path.join(os.getcwd(), "checkpoints"),
                    exist_ok=True)
        
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, 'checkpoints/best_model.pth')