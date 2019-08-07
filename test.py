import yaml
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from model import RecurrentTagger
from utils import ToweDataset, sentence_clip, eval, show
from sklearn.metrics.scorer import precision_score, recall_score, f1_score
from visualizer import Visualizer

config = yaml.load(open('config.yml'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

base_path = os.path.join('./data/', config['dataset'])
test_data_path = os.path.join(base_path, 'test.npz')
log_path = os.path.join(base_path, 'log.yml')
glove_path = os.path.join(base_path, 'glove.npy')
save_path = os.path.join(base_path, 'tagger.pkl')

test_data = ToweDataset(test_data_path)

test_loader = DataLoader(
    dataset=test_data,
    batch_size=config['batch_size'],
    shuffle=False,
    pin_memory=True
)

# log = yaml.load(open(log_path))
#
# embedding = nn.Embedding(
#     num_embeddings=log['vocab_size'],
#     embedding_dim=config['embed_size']
# )
# embedding.weight.data.copy_(torch.from_numpy(np.load(glove_path)))
#
# tagger = RecurrentTagger(
#     embedding=embedding,
#     bio_embed_size=config['bio_embed_size'],
#     hidden_size=config['hidden_size'],
#     num_layers=config['num_layers'],
#     dropout=config['dropout'],
#     bidirectional=config['bidirectional']
# )

tagger = torch.load(save_path)

tagger = tagger.cuda()

criterion = nn.CrossEntropyLoss(ignore_index=-1)

_, test_precision, test_recall, test_f1_score = eval(tagger, test_loader, criterion)

print('test precision: %.4f\trecall: %.4f\tf1_score: %.4f' % (test_precision, test_recall, test_f1_score))