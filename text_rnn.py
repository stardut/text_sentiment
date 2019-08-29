import torch
import torch.nn as nn
import torch.nn.functional as F


class TextRnn(object):
  """docstring for TextRnn"""

  def __init__(self, seq_len, num_embeding, hidden_size, num_layers):
    super(TextRnn, self).__init__()
    self.rnn = nn.LSTM(input_size=num_embeding,
                       hidden_size=hidden_size,
                       num_layers=num_layers,
                       batch_first=True)
    self.embedding = nn.Embedding(seq_len, num_embeding)
    self.fc = nn.Linear(hidden_size, 4)

  def forward(x):
    x = self.embedding(x)  # batch_size, seq_len, embedding_num
    out, _ = self.rnn(x, None)  # out: batch_size, seq_len, hidden_size
    out = self.fc(out[:, -1, :])
    return out
