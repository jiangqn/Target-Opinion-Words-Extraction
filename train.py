import yaml
import os
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
from model import ConvTagger, RecurrentTagger, FocalLoss
from utils import ToweDataset, sentence_clip, eval, show, error_analysis
from sklearn.metrics.scorer import precision_score, recall_score, f1_score
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
save_path = os.path.join(base_path, 'tagger.pkl')

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

# tagger = ConvTagger(
#     embedding=embedding,
#     bio_embed_size=config['bio_embed_size'],
#     kernel_sizes=config['kernel_sizes'],
#     kernel_num=config['kernel_num'],
#     num_layers=config['num_layers'],
#     dropout=config['dropout']
# )

tagger = RecurrentTagger(
    embedding=embedding,
    bio_embed_size=config['bio_embed_size'],
    hidden_size=config['hidden_size'],
    num_layers=config['num_layers'],
    dropout=config['dropout'],
    bidirectional=config['bidirectional']
)

tagger = tagger.cuda()

criterion = nn.CrossEntropyLoss(ignore_index=-1)
# criterion = FocalLoss(gamma=1.5, ignore_index=-1)
# criterion = BinaryFocalLoss(alpha=1, gamma=2, ignore_index=-1)

optimizer = optim.Adam(tagger.parameters(), lr=config['learning_rate'], weight_decay=config['l2_reg'])

visualizer = Visualizer(plot_path)

max_val_f1_score = 0

for epoch in range(config['num_epoches']):
    preds_collection = []
    labels_collection = []
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
        # error_analysis(sentences[0], targets[0], labels[0], tagger, index2word)
        logits = tagger(sentences, targets)
        logits = logits.view(-1, 2)
        preds = logits.argmax(dim=-1)
        labels = labels.view(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = (labels != -1).long().sum().item()
        total_loss += batch_size * loss.item()
        total_samples += batch_size
        preds_collection.append(preds)
        labels_collection.append(labels)
        # if i % 100 == 0:
        #     print('[epoch %d] [step %d] [loss %.4f]' % (epoch, i, loss.item()))
    preds_collection = torch.cat(preds_collection, dim=0)
    labels_collection = torch.cat(labels_collection, dim=0)
    mask = labels_collection != -1
    preds_collection = preds_collection.masked_select(mask).cpu().numpy()
    labels_collection = labels_collection.masked_select(mask).cpu().numpy()
    train_precision = precision_score(y_true=labels_collection, y_pred=preds_collection, labels=[0, 1], average='macro')
    train_recall = recall_score(y_true=labels_collection, y_pred=preds_collection, labels=[0, 1], average='macro')
    train_f1 = f1_score(y_true=labels_collection, y_pred=preds_collection, labels=[0, 1], average='macro')
    train_loss = total_loss / total_samples
    print('epoch: %d\ttrain\tloss: %.4f\tprecision: %.4f\trecall: %.4f\tf1: %.4f' % (epoch, train_loss, train_precision, train_recall, train_f1))
    val_loss, val_precision, val_recall, val_f1 = eval(tagger, val_loader, criterion)
    print('epoch: %d\tval\tloss: %.4f\tprecision: %.4f\trecall: %.4f\tf1: %.4f\n' % (epoch, val_loss, val_precision, val_recall, val_f1))
    visualizer.add(epoch, train_loss, train_precision, train_recall, train_f1, val_loss, val_precision, val_recall, val_f1)
    if val_f1 > max_val_f1_score:
        max_val_f1_score = val_f1
        torch.save(tagger, save_path)
        print('save model')

print('max val f1_score: %.4f' % max_val_f1_score)

visualizer.plot()