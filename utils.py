import operator
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from sklearn.metrics.scorer import precision_score, recall_score, f1_score

PAD = '<pad>'
UNK = '<unk>'
PAD_INDEX = 0
UNK_INDEX = 1

def sentence_clip(sentence):
    mask = (sentence != PAD_INDEX)
    sentence_lens = mask.long().sum(dim=1, keepdim=False)
    max_len = sentence_lens.max().item()
    return sentence[:, :max_len]

class Vocab(object):

    def __init__(self):
        self.count_dict = dict()
        self.predefined_list = [PAD, UNK]

    def add(self, word):
        if word in self.count_dict:
            self.count_dict[word] += 1
        else:
            self.count_dict[word] = 1

    def add_list(self, words):
        for word in words:
            self.add(word)

    def get_vocab(self, max_size=None, min_freq=0):
        sorted_words = sorted(self.count_dict.items(), key=operator.itemgetter(1), reverse=True)
        word2index = {}
        for word in self.predefined_list:
            word2index[word] = len(word2index)
        for word, freq in sorted_words:
            if (max_size is not None and len(word2index) >= max_size) or freq < min_freq:
                word2index[word] = word2index[UNK]
            else:
                word2index[word] = len(word2index)
        index2word = {}
        index2word[word2index[UNK]] = UNK
        for word, index in word2index.items():
            if index == word2index[UNK]:
                continue
            else:
                index2word[index] = word
        return word2index, index2word

BIOdict = {'O': 0, 'B': 1, 'I': 1}

def list_map(L, d):
    f = lambda x: d[x] if x in d else 0
    return [f(x) for x in L]

def data_transform(text_data, word2index, path):
    sentences = []
    targets = []
    labels = []
    max_len = 0
    for line in text_data:
        sentence, target, label = line.strip().split('\t')[1:]
        sentence = sentence.split(' ')
        target = target.split(' ')
        label = label.split(' ')
        target = [x.split('\\')[1] for x in target]
        label = [x.split('\\')[1] for x in label]
        assert len(sentence) == len(target) and len(sentence) == len(label)
        sentences.append(list_map(sentence, word2index))
        targets.append(list_map(target, BIOdict))
        labels.append(list_map(label, BIOdict))
        max_len = max(max_len, len(sentence))
    num = len(sentences)
    for i in range(num):
        sentences[i] = sentences[i] + [0] * (max_len - len(sentences[i]))
        targets[i] = targets[i] + [0] * (max_len - len(targets[i]))
        labels[i] = labels[i] + [-1] * (max_len - len(labels[i]))
    sentences = np.asarray(sentences, dtype=np.int32)
    targets = np.asarray(targets, dtype=np.int32)
    labels = np.asarray(labels, dtype=np.int32)
    np.savez(path, sentences=sentences, targets=targets, labels=labels)

def load_glove(path, vocab_size, word2index):
    if not os.path.isfile(path):
        raise IOError('Not a file', path)
    glove = np.random.uniform(-0.01, 0.01, [vocab_size, 300])
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            content = line.split(' ')
            if content[0] in word2index:
                glove[word2index[content[0]]] = np.array(list(map(float, content[1:])))
    glove[PAD_INDEX, :] = 0
    return glove

class ToweDataset(Dataset):

    def __init__(self, path):
        super(ToweDataset, self).__init__()
        data = np.load(path)
        self.sentences = torch.from_numpy(data['sentences']).long()
        self.targets = torch.from_numpy(data['targets']).long()
        self.labels = torch.from_numpy(data['labels']).long()
        self.len = self.sentences.size(0)

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        return self.sentences[item], self.targets[item], self.labels[item]

def eval(tagger, data_loader, criterion):
    tagger.eval()
    preds_collection = []
    labels_collection = []
    total_loss = 0
    total_samples = 0
    for i, data in enumerate(data_loader):
        sentences, targets, labels = data
        sentences, targets, labels = sentences.cuda(), targets.cuda(), labels.cuda()
        sentences = sentence_clip(sentences)
        targets = targets[:, 0:sentences.size(1)].contiguous()
        labels = labels[:, 0:sentences.size(1)].contiguous()
        logits = tagger(sentences, targets)
        logits = logits.view(-1, 2)
        preds = logits.argmax(dim=-1)
        labels = labels.view(-1)
        loss = criterion(logits, labels).item()
        batch_size = (labels != -1).long().sum().item()
        total_loss += loss * batch_size
        total_samples += batch_size
        preds_collection.append(preds)
        labels_collection.append(labels)
    preds_collection = torch.cat(preds_collection, dim=0)
    labels_collection = torch.cat(labels_collection, dim=0)
    mask = labels_collection != -1
    preds_collection = preds_collection.masked_select(mask).cpu().numpy()
    labels_collection = labels_collection.masked_select(mask).cpu().numpy()
    precision = precision_score(y_true=labels_collection, y_pred=preds_collection, labels=[0, 1], average='macro')
    recall = recall_score(y_true=labels_collection, y_pred=preds_collection, labels=[0, 1], average='macro')
    f1 = f1_score(y_true=labels_collection, y_pred=preds_collection, labels=[0, 1], average='macro')
    loss = total_loss / total_samples
    return loss, precision, recall, f1

def show(sentence, target, label, index2word):
    sentence, target, label = sentence.tolist(), target.tolist(), label.tolist()
    length = min([x if label[x]==-1 else 1000 for x in range(len(label))])
    text = ''.join([index2word[x] + ' ' for x in sentence[0:length]])
    print(text)
    print(sentence[0:length])
    print(target[0:length])
    print(label[0:length])

def error_analysis(sentence, target, label, model, index2word):
    pred = model(sentence.unsqueeze(0), target.unsqueeze(0)).squeeze(0).argmax(dim=-1).tolist()
    sentence, target, label = sentence.tolist(), target.tolist(), label.tolist()
    length = min([x if label[x]==-1 else 1000 for x in range(len(label))])
    text = ''.join([index2word[x] + ' ' for x in sentence[0:length]])
    print(text)
    print(sentence[0:length])
    print(target[0:length])
    print(label[0:length])
    print(pred[0:length])