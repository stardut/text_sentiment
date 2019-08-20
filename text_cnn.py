import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    """docstring for Net"""

    def __init__(self, vocab_size, window_size, seq_len):
        super(Net, self).__init__()
        embedding_size = 256
        feature_size = 128
        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(embedding_size, feature_size, w_size),
                nn.ReLU(),
                nn.MaxPool1d(seq_len - w_size + 1)) for w_size in window_size
        ])
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=embedding_size)
        self.fc1 = nn.Linear(feature_size * len(window_size), 128)
        self.fc2 = nn.Linear(128, 4)

    def forward(self, x):
        x = self.embedding(x)  # b * len * embedding
        x = x.permute(0, 2, 1)  # b * embedding * len
        x = [conv(x) for conv in self.conv]
        x = torch.cat(x, dim=1)  # b * (feature*window_size) * 1
        x = x.view(-1, x.size(1))  # b * (feature*window_size)
        x = self.fc1(x)  # b * 128
        x = F.dropout(x, 0.5, self.training)
        y = self.fc2(x)
        return y
