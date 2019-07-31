import torch
from torch import nn
from utils import sentence_clip

class ConvTagger(nn.Module):

    def __init__(self, embedding):
        super(ConvTagger).__init__()

    def forward(self, sentences, targets):
        sentences = sentence_clip(sentences)
        targets = targets[:, sentences.size(1)].contiguous()
