# import numpy as np
# import gensim
# import pandas as pd
# from preprocessing import Preprocesser

# # read the input files
# def read_files(file_path: str,
#                label: int):
#     prep = Preprocesser()
#     df = pd.DataFrame()
    
#     with open(file_path, errors="ignore") as f:
#         for line in f:
#             sentence = line.strip()
#             cleaned_sentence = prep(str(sentence))
#             temp = pd.DataFrame({'sentence': [cleaned_sentence],
#                                  'label': [label]})
#             df = pd.concat([df, temp], axis=0)
#             del temp
            
#     return df

# # creates the vocabulary dict
# # will be used to tokenize the data and to create the embedding
# def create_vocab(sentences: list) -> dict:
#     word_list = " ".join(sentences).split()
#     word_list = list(set(word_list))
#     word_list.sort() # making sure that we create the vocab in the same order every time we run
#     word_dict = {w: i+1 for i, w in enumerate(word_list)}
#     return word_dict

# # prepares the data
# # (tokenize, truncate and pad)
# def prepare_data(input: str,
#                  vocab: dict,
#                  max_len: int) -> np.array:
#     if len(input.split()) >= max_len:
#         input = " ".join(input.split()[:max_len])
#         tokenized_inputs = np.array([vocab[str(word)] for word in input.split()])
#     else:
#         input = [vocab[str(word)] for word in input.split()]
#         input += [0] * (max_len - len(input))
#         tokenized_inputs = np.array(input)
#     return tokenized_inputs

# # loads a pre trained embedding
# # e.g: glove/word2vec
# def load_pretrained_embedding(path: str,
#                               vocab: dict,
#                               embedding_dim: int) -> np.array:
#     model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
#     embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))
    
#     for word in vocab.keys():
#         try:
#             embedding_vector = model[word]
#             if embedding_vector is not None:
#                 embedding_matrix[vocab[word]] = embedding_vector
#             else:
#                 embedding_matrix[vocab[word]] = np.random.randn(embedding_dim)
#         except KeyError:
#             embedding_matrix[vocab[word]] = np.random.randn(embedding_dim)
    
#     return embedding_matrix