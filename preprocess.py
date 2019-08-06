import yaml
import os
import pickle
import numpy as np
import random
from utils import Vocab, data_transform, load_glove

# load config

config = yaml.load(open('config.yml'))

# build paths

base_path = os.path.join('./data/', config['dataset'])
train_src_path = os.path.join(base_path, 'train.tsv')
test_src_path = os.path.join(base_path, 'test.tsv')
train_trg_path = os.path.join(base_path, 'train.npz')
val_trg_path = os.path.join(base_path, 'val.npz')
test_trg_path = os.path.join(base_path, 'test.npz')
word2index_path = os.path.join(base_path, 'word2index.pickle')
index2word_path = os.path.join(base_path, 'index2word.pickle')
glove_path = os.path.join(base_path, 'glove.npy')
log_path = os.path.join(base_path, 'log.yml')

# load training and test data

train_data = open(train_src_path, 'r', encoding='utf-8').readlines()[1:]
test_data = open(test_src_path, 'r', encoding='utf-8').readlines()[1:]

# split training and validation datasets

random.shuffle(train_data)

num = len(train_data)
val_num = int(num * config['val_rate'])
val_data = train_data[num - val_num: num]
train_data = train_data[0: num - val_num]

# build vocabulary

vocab = Vocab()

for line in train_data:
    line = line.strip().split('\t')[1].split(' ')
    vocab.add_list(line)

word2index, index2word = vocab.get_vocab(max_size=config['max_size'], min_freq=config['min_freq'])

vocab_size = len(index2word)
oov_size = len(word2index) - len(index2word)

with open(word2index_path, 'wb') as handle:
    pickle.dump(word2index, handle)
with open(index2word_path, 'wb') as handle:
    pickle.dump(index2word, handle)

glove = load_glove(config['glove_path'], vocab_size, word2index)
np.save(glove_path, glove)

# transform data

data_transform(train_data, word2index, train_trg_path)
data_transform(val_data, word2index, val_trg_path)
data_transform(test_data, word2index, test_trg_path)

# save log

log = {
        'vocab_size': vocab_size,
        'oov_size': oov_size,
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data)
    }
with open(log_path, 'w') as handle:
    yaml.safe_dump(log, handle, encoding='utf-8', allow_unicode=True, default_flow_style=False)