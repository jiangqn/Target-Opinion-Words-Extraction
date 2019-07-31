import operator
import numpy as np
import os

PAD = '<pad>'
UNK = '<unk>'
PAD_INDEX = 0
UNK_INDEX = 1

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

BIOdict = {'O': 0, 'B': 1, 'I': 2}

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
        sentences[i] = sentences[i] + [0] * (num - len(sentences[i]))
        targets[i] = targets[i] + [0] * (num - len(targets[i]))
        labels[i] = labels[i] + [0] * (num - len(labels[i]))
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