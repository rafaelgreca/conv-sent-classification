# Convolutional Neural Networks for Sentence Classification

It is slightly simplified implementation of Yoon Kim's [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) paper in PyTorch.

Use of this code may be cited as follows:

```
@article{DBLP:journals/corr/Kim14f,
  author    = {Yoon Kim},
  title     = {Convolutional Neural Networks for Sentence Classification},
  journal   = {CoRR},
  volume    = {abs/1408.5882},
  year      = {2014},
  url       = {http://arxiv.org/abs/1408.5882},
  eprinttype = {arXiv},
  eprint    = {1408.5882},
  timestamp = {Mon, 13 Aug 2018 16:46:21 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/Kim14f.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Installation

To install this package, clone the repository from GitHub to a directory of your choice and install using pip:
```bash
git clone https://github.com/rafaelgreca/conv-sent-classification.git
```

You need to create a conva environment using conda and install the requirements:
```bash
conda create -n venv python=3.8.10
conda activate venv
pip install -r requirements.txt
```

## How to Train

To run the code is very straight forward, you just need to do:

```python3 
python3 main.py --model "rand"
```

All parameters available to use:
- `--model`: Which model you want to build ("rand", "static" or "non-static");
- `--batch_size`: The batch size of the training step (optional). Default: 50;
- `--epochs`: How many epochs you want to train the model (optional). Default: 25;
- `--max_len`: The maximum length of each sentence (optional). Default: 30.

## License

[MIT](https://choosealicense.com/licenses/mit/)
