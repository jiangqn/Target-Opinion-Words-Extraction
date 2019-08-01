import yaml
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from model import ConvTagger
from utils import ToweDataset

config = yaml.load(open('config.yml'))

base_path = os.path.join('./data/', config['dataset'])
train_data_path = os.path.join(base_path, 'train.npz')
val_data_path = os.path.join(base_path, 'val.npz')
log_path = os.path.join(base_path, 'log.yml')

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

tagger = ConvTagger(
    embedding=embedding,
    bio_embed_size=config['bio_embed_size'],
    kernel_sizes=config['kernel_sizes'],
    kernel_num=config['kernel_num'],
    dropout=config['dropout']
)

tagger = tagger.cuda()

criterion = nn.CrossEntropyLoss(ignore_index=-1)

optimizer = optim.Adam(tagger.parameters(), lr=config['learning_rate'])

for epoch in range(config['num_epoches']):
    for i, data in enumerate(train_loader):
        sentences, targets, labels = data
        sentences, targets, labels = sentences.cuda(), targets.cuda(), labels.cuda()
        logits = tagger(sentences, targets)
        logits = logits.view(-1, 3)
        labels = labels.view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, loss.item()))