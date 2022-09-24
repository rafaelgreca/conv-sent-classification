import re


def preprocessor(sentence: str) -> str:
    """
    Preprocessing step inspired in the original code implemented by Yoon Kim.
    You can check it out here: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py.
    """
    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    sentence = re.sub(r"\'s", " 's", sentence)
    sentence = re.sub(r"\'ve", " 've", sentence)
    sentence = re.sub(r"n\'t", " n't", sentence)
    sentence = re.sub(r"\'re", " 're", sentence)
    sentence = re.sub(r"\'d", " 'd", sentence)
    sentence = re.sub(r"\'ll", " 'll", sentence)
    sentence = re.sub(r",", " , ", sentence)
    sentence = re.sub(r"!", " ! ", sentence)
    sentence = re.sub(r"\(", " \( ", sentence)
    sentence = re.sub(r"\)", " \) ", sentence)
    sentence = re.sub(r"\?", " \? ", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.strip().lower()


def preprocessor_sst(sentence: str) -> str:
    """
    Tokenization/string cleaning for the SST dataset
    """
    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.strip().lower()
