import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import dataset
from text_cnn import Net as text_cnn

max_len = 30
window_size = [1, 2, 3, 4, 5] * 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_data = dataset('data/train.csv', max_len=max_len)
iter_train = train_data.next(2)
# test_data = dataset('data/test.csv', max_len=max_len)
# iter_test = test_data.next()


vocab_len = train_data.vocab_len
net = text_cnn(vocab_len, window_size, max_len)

lr = 0.01
op = optim.Adam(net.parameters(), lr)
net.to(device)


for step in range(1000):
    inputs, labels = next(iter_train)
    labels = torch.from_numpy(labels)
    inputs = torch.LongTensor(inputs)
    inputs = inputs.to(device)
    target = labels.to(device)

    op.zero_grad()
    output = net.forward(inputs)

    loss = F.cross_entropy(output, target)
    loss.backward()
    op.step()

    if step % 100 == 99:
        print('step: %4d loss: %.3f' % (step + 1, loss.item()))
        print(target)
        print(output)
