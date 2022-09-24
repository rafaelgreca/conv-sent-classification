import os
import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Any


class CNN(nn.Module):
    """
    Yoon Kim"s Convolutional Neural Network (CNN) models implementation described
    in https://arxiv.org/pdf/1408.5882.pdf. Models implemented so far: embedding randomly initialized
    (rand), embedding initialized with pretrained embedding's weights and not trainable (static) and 
    embedding initialized with pretrained embedding's weights and trainable (non-static).
    """

    def __init__(
        self,
        model: str,
        vocab_size: int,
        embedding_dim: int,
        pad_idx: Any,
        pretrained_embedding: Any,
    ) -> None:
        """
        :param model: which model will be created ("rand", "static" or "non-static").
        :param vocab_size: the vocabulary length.
        :param embedding dim: the embedding dimmension.
        :param embedding_weights: the pretrained embedding weights (for the "static" and "non-static" model).
        """
        super(CNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)

        if model == "static":
            self.embedding = self.embedding.from_pretrained(pretrained_embedding)
            self.embedding.requires_grad = False
        elif model == "non-static":
            self.embedding = self.embedding.from_pretrained(pretrained_embedding)

        # create the convolutional layers
        self.conv1 = nn.Conv2d(1, 100, (3, embedding_dim))
        self.conv2 = nn.Conv2d(1, 100, (4, embedding_dim))
        self.conv3 = nn.Conv2d(1, 100, (5, embedding_dim))

        # create the dropout layer
        self.dropout = nn.Dropout(0.5)

        # create the dense/fully connected layer (output layer)
        self.fc = nn.Linear(300, 1)

    def forward(self, text: torch.tensor) -> torch.tensor:
        # Embedding Layer
        embedded = self.embedding(text)  # (batch size, sentence_len, embedding_dim)

        embedded = embedded.unsqueeze(1)  # (batch size, 1, sentence_len, embedding_dim)

        # Convolutional layers
        output_conv1 = F.relu(self.conv1(embedded).squeeze(3))
        output_conv2 = F.relu(self.conv2(embedded).squeeze(3))
        output_conv3 = F.relu(self.conv3(embedded).squeeze(3))

        # Max pooling layers
        output_maxpool1 = F.max_pool1d(output_conv1, output_conv1.size(2)).squeeze(2)
        output_maxpool2 = F.max_pool1d(output_conv2, output_conv2.size(2)).squeeze(2)
        output_maxpool3 = F.max_pool1d(output_conv3, output_conv3.size(2)).squeeze(2)

        output_maxpool = torch.cat(
            (output_maxpool1, output_maxpool2, output_maxpool3), dim=1
        )

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

    def __init__(self) -> None:
        self.best_valid_loss = float("inf")
        self.best_valid_accuracy = 0.0
        os.makedirs(os.path.join(os.getcwd(), "checkpoints"), exist_ok=True)

    def __call__(
        self,
        current_valid_loss: float,
        current_valid_accuracy: float,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim,
        criterion: torch.nn.BCEWithLogitsLoss,
    ) -> None:
        if current_valid_accuracy > self.best_valid_accuracy:
            self.best_valid_loss = current_valid_loss
            self.best_valid_accuracy = current_valid_accuracy
            print("\nSaving model...")
            print(f"Epoch: {epoch}")
            print(f"Best validation loss: {self.best_valid_loss:1.6f}")
            print(f"Best validation accuracy: {self.best_valid_accuracy:1.6f}")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": criterion,
                },
                "checkpoints/best_model.pth",
            )
