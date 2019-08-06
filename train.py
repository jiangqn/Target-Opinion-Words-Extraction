import yaml
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from model import ConvTagger
from utils import ToweDataset, sentence_clip, eval, show
from visualizer import Visualizer

config = yaml.load(open('config.yml'))
os.environ['CUDA_VISIBLE_DEVICES'] = str(config['gpu'])

base_path = os.path.join('./data/', config['dataset'])
train_data_path = os.path.join(base_path, 'train.npz')
val_data_path = os.path.join(base_path, 'val.npz')
log_path = os.path.join(base_path, 'log.yml')
glove_path = os.path.join(base_path, 'glove.npy')
plot_path = './plots/training.jpg'
index2word_path = os.path.join(base_path, 'index2word.pickle')

with open(index2word_path, 'rb') as handle:
    index2word = pickle.load(handle)

train_data = ToweDataset(train_data_path)
val_data = ToweDataset(val_data_path)

train_loader = DataLoader(
    dataset=train_data,
    batch_size=config['batch_size'],
    shuffle=True,
    pin_memory=True
)
val_loader = DataLoader(
    dataset=val_data,
    batch_size=config['batch_size'],
    shuffle=False,
    pin_memory=True
)

log = yaml.load(open(log_path))

embedding = nn.Embedding(
    num_embeddings=log['vocab_size'],
    embedding_dim=config['embed_size']
)
embedding.weight.data.copy_(torch.from_numpy(np.load(glove_path)))

tagger = ConvTagger(
    embedding=embedding,
    bio_embed_size=config['bio_embed_size'],
    kernel_sizes=config['kernel_sizes'],
    kernel_num=config['kernel_num'],
    num_layers=config['num_layers'],
    dropout=config['dropout']
)

tagger = tagger.cuda()

criterion = nn.CrossEntropyLoss(ignore_index=-1)

optimizer = optim.Adam(tagger.parameters(), lr=config['learning_rate'])

visualizer = Visualizer(['epoch', 'train_loss', 'val_loss', 'precision', 'recall', 'f1'], plot_path)

for epoch in range(config['num_epoches']):
    total_loss = 0
    total_samples = 0
    for i, data in enumerate(train_loader):
        tagger.train()
        optimizer.zero_grad()
        sentences, targets, labels = data
        sentences, targets, labels = sentences.cuda(), targets.cuda(), labels.cuda()
        sentences = sentence_clip(sentences)
        targets = targets[:, 0:sentences.size(1)].contiguous()
        labels = labels[:, 0:sentences.size(1)].contiguous()
        logits = tagger(sentences, targets)
        logits = logits.view(-1, 3)
        labels = labels.view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = (labels != -1).long().sum().item()
        total_loss += batch_size * loss.item()
        total_samples += batch_size
        # if i % 100 == 0:
        #     print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, loss.item()))
    train_loss = total_loss / total_samples
    val_loss, precision, recall, f1 = eval(tagger, val_loader, criterion)
    print('train_loss: %.4f\tval_loss: %.4f\tprecision: %.4f\trecall: %.4f\tf1: %.4f' % (train_loss, val_loss, precision, recall, f1))
    visualizer.add([epoch, train_loss, val_loss, precision, recall, f1])

visualizer.plot()