import torch
from torch import nn
import torch.nn.functional as F
from utils import sentence_clip, PAD_INDEX

class ConvTagger(nn.Module):

    def __init__(self, embedding, bio_embed_size, kernel_sizes, kernel_num, dropout):
        super(ConvTagger, self).__init__()
        self.embed_size = embedding.embedding_dim
        self.embedding = embedding
        self.bio_embed_size = bio_embed_size
        self.bio_embedding = nn.Embedding(2, bio_embed_size)
        self.kernel_sizes = kernel_sizes
        self.kernels = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.kernels.append(nn.Conv1d(
                in_channels=self.embed_size + self.bio_embed_size,
                out_channels=kernel_num,
                kernel_size=kernel_size
            ))
        self.rep_size = sum(self.kernel_sizes) * kernel_num
        self.dropout = dropout
        self.output_projection = nn.Linear(self.rep_size, 3)

    def forward(self, sentences, targets):
        sentences = sentence_clip(sentences)
        targets = targets[:, 0:sentences.size(1)].contiguous()
        targets = torch.max(targets, torch.tensor(1).long().to(targets.device))
        mask = sentences != PAD_INDEX
        sentences = self.embedding(sentences)
        targets = self.bio_embedding(targets)
        sentences = torch.cat((sentences, targets), dim=-1).transpose(1, 2)
        feature_map = []
        for kernel_size, conv in zip(self.kernel_sizes, self.kernels):
            left_pad = (kernel_size - 1) // 2
            right_pad = kernel_size // 2
            feature_map.append(conv(F.pad(sentences, [left_pad, right_pad, 0, 0, 0, 0])))
        feature_map = torch.cat(feature_map, dim=1).transpose(1, 2)
        feature_map = torch.relu(feature_map)
        logits = self.output_projection(feature_map)
        logits = logits.masked_fill(mask.unsqueeze(-1)==0, 0)
        return logits