import argparse
import sys
import torch
import random
import os
import numpy as np
import warnings
import pandas as pd
import time
from preprocessing import Preprocesser
from utils import create_vocab, prepare_data, load_pretrained_embedding, create_data_loader, train_model, read_files
from sklearn.model_selection import train_test_split
from models import CNN

warnings.filterwarnings("ignore")
seed = 2109
device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
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
    parser.add_argument("-model", "--model", type=str, help="Which model use (rand, static, non-static)", required=True)
    parser.add_argument("-batch_size", "--batch_size", type=int, help="Batch size for training and evaluation", default=50)
    parser.add_argument("-embedding_path", "--embedding_path", type=str, help="Embedding file path", default="")
    parser.add_argument("-embedding_dim", "--embedding_dim", type=int, help="Embedding dimmension", default=300)
    parser.add_argument("-epochs", "--epochs", type=int, help="Epochs in training step for the CNN model", default=10)
    parser.add_argument("-max_len", "--max_len", type=int, help="Sequence max length", default=30)
    parser.add_argument("-test_size", "--test_size", type=float, help="Test size (percentage)", default=0.1)
    parser.add_argument("-dropout_rate", "--dropout_rate", type=float, help="Dropout rate", default=0.5)
    parser.add_argument("-filter_units", "--filter_units", type=int, help="Filter units", default=100)
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
        
        # reading the files
        pos_sentences = read_files(file_path=os.path.join(os.getcwd(), "data", "rt-polarity.pos"), label=1)
        neg_sentences = read_files(file_path=os.path.join(os.getcwd(), "data", "rt-polarity.neg"), label=0)
        all_df = pd.concat([pos_sentences, neg_sentences], axis=0)
        all_df = all_df.drop_duplicates()
        
        # getting all the sentences from the data
        # which will be used to create the vocabulary dict for the embedding layer
        all_sentences = []
        all_sentences.extend(all_df["sentence"])
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

        train_dataloader = create_data_loader(data=X_train.values,
                                              target=y_train.values,
                                              batch_size=args.batch_size,
                                              worker_init=_init_fn,
                                              generator=generator)

        validation_dataloader = create_data_loader(data=X_validation.values,
                                                   target=y_validation.values,
                                                   batch_size=args.batch_size,
                                                   worker_init=_init_fn,
                                                   generator=generator)

        s_time = time.process_time()
        train_model(model=model,
                    epochs=args.epochs,
                    train_dataloader=train_dataloader,
                    validation_dataloader=validation_dataloader,
                    device=device)
        e_time = time.process_time()
        print(f"{model.name} trained in {e_time-s_time:1.2f}s")