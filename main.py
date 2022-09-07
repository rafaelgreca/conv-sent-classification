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

# trains the model
def train(model: torch.nn.Module,
          train_dataloader: DataLoader,
          device: torch.device,
          optimizer: torch.optim,
          loss: nn.BCEWithLogitsLoss):
    train_loss = 0
    train_acc = 0
    model.train()
    
    for batch_idx, batch in enumerate(train_dataloader):
        data, target = batch["X"].to(device), batch["y"].type(torch.FloatTensor).to(device)
        optimizer.zero_grad()

        output = model(data).squeeze(1)
        l = loss(output, target)
        l.backward()
        optimizer.step()
        
        train_loss += l.item()
        output_pred = torch.round(torch.sigmoid(output))
        train_acc = output_pred.eq(target).sum().item() #convert into float for division 
        # acc = correct.sum() / len(correct)
    
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_dataloader.dataset),
                100. * batch_idx / len(train_dataloader), l.item()))
    
    train_loss /= len(train_dataloader.dataset)
    train_acc /= len(train_dataloader.dataset)
    print(train_acc)
    return train_loss, train_acc

def validation(model: torch.nn.Module,
               device: torch.device,
               validation_dataloder: DataLoader):
    validation_loss = 0
    validation_acc = 0
    model.eval()
    
    with torch.no_grad():
        for batch in validation_dataloder:
            data, target = batch["X"].to(device), batch["y"].type(torch.FloatTensor).to(device)
            output = model(data).squeeze(1)
            validation_loss += loss(output, target).item()  # sum up batch loss

        validation_loss /= len(validation_dataloder)
    return validation_loss, validation_acc
            
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
        assert os.path.exists(os.path.join(os.getcwd(), "data")), "You must create a folder called data with the train and test files inside"

        model_name = args.model
        
        # TODO: Refazer etapa de limpeza/pré-processamento
        # reading the files
        pos_sentences = read_files(file_path=os.path.join(os.getcwd(), "data", "rt-polarity.pos"), label=1)
        neg_sentences = read_files(file_path=os.path.join(os.getcwd(), "data", "rt-polarity.neg"), label=0)
        all_df = pd.concat([pos_sentences, neg_sentences], axis=0).reset_index(drop=True)
        all_df = all_df.drop_duplicates()
        
        # getting all the sentences from the data
        # which will be used to create the vocabulary dict for the embedding layer
        all_sentences = []
        all_sentences.extend(all_df["sentence"])
        vocab = create_vocab(all_sentences)

        # tokenizing and padding the train/test data
        all_df["tokenized_sentence"] = all_df["sentence"].apply(lambda x: prepare_data(x, vocab, args.max_len))

        X = all_df["tokenized_sentence"]
        y = all_df["label"]
        
        # splitting the training data into training and
        # validation data
        X_train, X_validation, y_train, y_validation = train_test_split(X,
                                                                        y,
                                                                        test_size=args.test_size,
                                                                        random_state=seed,
                                                                        shuffle=False)
        
        if model_name != "rand":

            if args.embedding_path == "":
                embedding_path = os.path.join(os.getcwd(), "pretrained_embedding", "GoogleNews-vectors-negative300.bin.gz")
            else:
                embedding_path = args.embedding_path

            # saving a pretrained embedding to use in the
            # yoon kim"s cnn-static or cnn-non-static model
            embedding_weights = load_pretrained_embedding(path=embedding_path,
                                                          vocab=vocab,
                                                          embedding_dim=args.embedding_dim)
        else:
            embedding_weights = None

        model = CNN(vocab_size=len(vocab)+1,
                    model=model_name,
                    embedding_dim=args.embedding_dim,
                    embedding_weights=embedding_weights,
                    n_filters=args.filter_units,
                    filter_sizes=[3, 4, 5],
                    output_dim=1,
                    dropout=args.dropout_rate).to(device)

        model.name = model_name

        train_dataset = DatasetDL(data=pd.DataFrame({"X": X_train.values.tolist(),
                                                     "y": y_train.values.tolist()}))
        
        train_dataloader = DataLoader(train_dataset,
                                      batch_size=args.batch_size,
                                      num_workers=0,
                                      worker_init_fn=_init_fn,
                                      generator=generator,
                                      shuffle=False)

        validation_dataset = DatasetDL(data=pd.DataFrame({"X": X_validation.values.tolist(),
                                                          "y": y_validation.values.tolist()}))
        
        validation_dataloader = DataLoader(validation_dataset,
                                           batch_size=args.batch_size,
                                           num_workers=0,
                                           worker_init_fn=_init_fn,
                                           generator=generator,
                                           shuffle=False)
    
        optimizer = torch.optim.Adam(model.parameters())
        loss = nn.BCEWithLogitsLoss()
        save_best_model = SaveBestModel()
        
        s_time = time.process_time()
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train(model=model,
                                          train_dataloader=train_dataloader,
                                          device=device,
                                          optimizer=optimizer,
                                          loss=loss)
            validation_loss, validation_acc = validation(model=model,
                                                         device=device,
                                                         validation_dataloder=validation_dataloader)

            save_best_model(current_valid_loss=validation_loss,
                            epoch=epoch,
                            model=model,
                            optimizer=optimizer,
                            criterion=loss)
            
        e_time = time.process_time()
        print(f"{model.name} trained in {e_time-s_time:1.2f}s")