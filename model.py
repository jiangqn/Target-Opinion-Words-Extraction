import torch
from torch import nn
import torch.nn.functional as F
from utils import PAD_INDEX

class ConvTagger(nn.Module):

    def __init__(self, embedding, bio_embed_size, kernel_sizes, kernel_num, num_layers, dropout):
        super(ConvTagger, self).__init__()
        self.embed_size = embedding.embedding_dim
        self.embedding = embedding
        self.bio_embed_size = bio_embed_size
        self.bio_embedding = nn.Embedding(2, bio_embed_size)
        self.kernel_sizes = kernel_sizes
        self.kernel_num = kernel_num
        self.num_layers = num_layers
        self.feature_size = len(self.kernel_sizes) * kernel_num
        self.dropout = dropout
        self.conv_layers = nn.ModuleList([
            ConvLayer(
                input_size=self.embed_size + self.bio_embed_size,
                kernel_sizes=self.kernel_sizes,
                kernel_num=self.kernel_num,
                dropout=self.dropout
            )
        ])
        for i in range(self.num_layers - 1):
            self.conv_layers.append(ConvLayer(
                input_size=self.feature_size,
                kernel_sizes=self.kernel_sizes,
                kernel_num=self.kernel_num,
                dropout=self.dropout
            ))
        self.output_projection = nn.Linear(self.feature_size, 2)

    def forward(self, sentences, targets):
        mask = sentences != PAD_INDEX
        sentences = self.embedding(sentences)
        targets = self.bio_embedding(targets)
        feature_map = torch.cat((sentences, targets), dim=-1).transpose(1, 2)
        for i, layer in enumerate(self.conv_layers):
            feature_map = layer(feature_map)
        feature_map = feature_map.transpose(1, 2)
        logits = self.output_projection(feature_map)
        logits = logits.masked_fill(mask.unsqueeze(-1)==0, 0)
        return logits

class ConvLayer(nn.Module):

    def __init__(self, input_size, kernel_sizes, kernel_num, dropout=0):
        super(ConvLayer, self).__init__()
        self.input_size = input_size
        self.output_size = len(kernel_sizes) * kernel_num
        self.kernel_sizes = kernel_sizes
        self.norm = nn.BatchNorm1d(input_size)
        self.kernels = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.kernels.append(nn.Conv1d(
                in_channels=input_size,
                out_channels=kernel_num,
                kernel_size=kernel_size
            ))
        self.dropout = dropout

    def forward(self, input):
        input = self.norm(input)
        feature_map = []
        for kernel_size, kernel in zip(self.kernel_sizes, self.kernels):
            left_pad = (kernel_size - 1) // 2
            right_pad = kernel_size // 2
            feature_map.append(torch.relu(kernel(F.pad(input, [left_pad, right_pad, 0, 0, 0, 0]))))
        feature_map = torch.cat(feature_map, dim=1)
        feature_map = F.dropout(feature_map)
        if input.size() == feature_map.size():
            feature_map = feature_map + input
        return feature_map

class RecurrentTagger(nn.Module):

    def __init__(self, embedding, bio_embed_size, hidden_size, num_layers, dropout, bidirectional):
        super(RecurrentTagger, self).__init__()
        self.embed_size = embedding.embedding_dim
        self.embedding = embedding
        self.bio_embed_size = bio_embed_size
        self.bio_embedding = nn.Embedding(2, bio_embed_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.rnn = nn.LSTM(
            input_size=self.embed_size + self.bio_embed_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional
        )
        self.feature_size = (2 if bidirectional else 1) * hidden_size
        self.output_projection = nn.Linear(self.feature_size, 2)

    def forward(self, sentences, targets):
        # mask = sentences != PAD_INDEX
        sentences = self.embedding(sentences)
        targets = self.bio_embedding(targets)
        feature_map = torch.cat((sentences, targets), dim=-1)
        feature_map, _ = self.rnn(feature_map)
        logits = self.output_projection(feature_map)
        # logits = logits.masked_fill(mask.unsqueeze(-1) == 0, 0)
        return logits


class FocalLoss(nn.Module):

    def __init__(self, gamma=2.0, ignore_index=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits, labels):
        if self.ignore_index != None:
            mask = labels != self.ignore_index
            labels = torch.max(labels, torch.tensor(0).to(labels.device))
        one_hot = torch.zeros_like(logits).to(logits.device)
        one_hot = one_hot.scatter(1, labels.unsqueeze(-1), 1)
        prob = torch.softmax(logits, dim=-1)
        loss = - one_hot * torch.log(prob) * (1 - prob).pow(self.gamma)
        loss = loss.sum(dim=1, keepdim=False)
        if self.ignore_index != None:
            loss = loss.masked_select(mask)
        loss = loss.mean()
        return loss