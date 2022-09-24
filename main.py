import argparse
import sys
from typing import Tuple
import torch
import random
import os
import numpy as np
import warnings
import pandas as pd
import time
import torch.nn as nn

from model import CNN, SaveBestModel
from torchtext.datasets import IMDB
from torchtext.data import Field, LabelField, BucketIterator, Pipeline
from preprocessing import preprocessor, preprocessor_sst

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
torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(
    model: torch.nn.Module,
    dataloader: BucketIterator,
    optimizer: torch.optim.Adam,
    loss: torch.nn.BCEWithLogitsLoss,
) -> Tuple[float, float]:
    model.train()
    training_loss = 0
    training_accuracy = 0

    for i, batch in enumerate(dataloader):
        data, label = batch.text, batch.label
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(data).squeeze(1)
        pred = torch.round(torch.sigmoid(output))

        l = loss(output, label)
        l.backward()
        optimizer.step()

        training_loss += l.item()
        training_accuracy += pred.eq(label).sum().item() / len(label)

    training_loss /= len(dataloader)
    training_accuracy /= len(dataloader)
    return training_loss, training_accuracy


def validation(
    model: torch.nn.Module, dataloader: BucketIterator, optimizer: torch.optim.Adam
) -> Tuple[float, float]:
    model.eval()
    validation_loss = 0
    validation_accuracy = 0

    for i, batch in enumerate(dataloader):
        data, label = batch.text, batch.label
        data, label = data.to(device), label.to(device)

        optimizer.zero_grad()
        output = model(data).squeeze(1)
        pred = torch.round(torch.sigmoid(output))

        l = loss(output, label)
        l.backward()
        optimizer.step()

        validation_loss += l.item()
        validation_accuracy += pred.eq(label).sum().item() / len(label)

    validation_loss /= len(dataloader)
    validation_accuracy /= len(dataloader)
    return validation_loss, validation_accuracy


def test(
    model: torch.nn.Module, dataloader: BucketIterator, loss: torch.nn.BCEWithLogitsLoss
) -> Tuple[float, float]:
    model.eval()
    test_loss = 0
    test_accuracy = 0

    for batch in dataloader:
        data, label = batch.text, batch.label
        data, label = data.to(device), label.to(device)

        output = model(data).squeeze(1)
        pred = torch.round(torch.sigmoid(output))

        l = loss(output, label)

        test_loss += l.item()
        test_accuracy += pred.eq(label).sum().item() / len(label)

    test_loss /= len(dataloader)
    test_accuracy /= len(dataloader)
    return test_loss, test_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        help="Which model use (rand, static, non-static)",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size for training and evaluation",
        default=50,
    )
    parser.add_argument(
        "--epochs",
        type=int,
        help="Epochs in training step for the CNN model",
        default=25,
    )
    parser.add_argument(
        "--max_len",
        type=int,
        help="Sequence max length",
        default=30
    )
    args = parser.parse_args()

    if not len(sys.argv) > 1:
        raise Exception("No arguments passed")
    else:
        available_models = ["rand", "static", "non-static"]

        assert (
            args.model in available_models
        ), f"You must provide a valid model. Valid models: {available_models}"

        model_name = args.model
        preprocessor_pipeline = Pipeline(preprocessor)

        text_field = Field(
            lower=True,
            batch_first=True,
            preprocessing=preprocessor_pipeline,
            fix_length=args.max_len,
        )
        label_field = LabelField(dtype=torch.float32)

        train_data, test_data = IMDB.splits(text_field, label_field)

        train_data, valid_data = train_data.split(random_state=random.seed(seed))

        if args.model == "rand":
            text_field.build_vocab(train_data)
            label_field.build_vocab(train_data)
            pad_idx = text_field.vocab.stoi[text_field.pad_token]
            unk_idx = text_field.vocab.stoi[text_field.unk_token]
            pretrained_embedding = None
        elif args.model == "static" or args.model == "non-static":
            text_field.build_vocab(train_data, vectors="glove.6B.300d")
            label_field.build_vocab(train_data)

            pad_idx = text_field.vocab.stoi[text_field.pad_token]
            unk_idx = text_field.vocab.stoi[text_field.unk_token]
            pretrained_embedding = text_field.vocab.vectors
            pretrained_embedding[unk_idx] = torch.rand(300)
            pretrained_embedding[pad_idx] = torch.rand(300)

        vocab_size = len(text_field.vocab)

        train_iter, dev_iter, test_iter = BucketIterator.splits(
            (train_data, valid_data, test_data), batch_size=args.batch_size
        )

        model = CNN(
            model="rand",
            vocab_size=vocab_size,
            embedding_dim=300,
            pad_idx=pad_idx,
            pretrained_embedding=pretrained_embedding,
        )
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters())
        loss = torch.nn.BCEWithLogitsLoss()
        save_best_model = SaveBestModel()

        for epoch in range(1, args.epochs + 1):
            print(f"\nTraining epoch: {epoch}")

            train_loss, train_acc = train(
                model=model, dataloader=train_iter, optimizer=optimizer, loss=loss
            )

            validation_loss, validation_accuracy = validation(
                model=model, dataloader=dev_iter, optimizer=optimizer
            )

            save_best_model(
                current_valid_accuracy=validation_accuracy,
                current_valid_loss=validation_loss,
                epoch=epoch,
                model=model,
                optimizer=optimizer,
                criterion=loss,
            )

        model.load_state_dict(
            torch.load("checkpoints/best_model.pth")["model_state_dict"]
        )

        test_loss, test_accuracy = test(model=model, dataloader=test_iter, loss=loss)

        print(f"\nTest loss: {test_loss:1.6f}")
        print(f"Test accuracy: {test_accuracy:1.6f}")
