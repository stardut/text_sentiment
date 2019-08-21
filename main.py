import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from data import dataset
from text_cnn import Net as text_cnn


class Worker(object):
    """docstring for Worker"""

    def __init__(self, vocab_len, max_len):
        super(Worker, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.max_len = max_len
        self.vocab_len = vocab_len
        self.net = self.text_cnn_net()
        self.net.to(self.device)

    def text_cnn_net(self):
        window_size = [2, 3, 4, 5] * 2
        net = text_cnn(self.vocab_len, window_size, self.max_len)
        return net

    def run_data(self, data):
        inputs, labels = data
        labels = torch.from_numpy(labels).to(self.device)
        inputs = torch.LongTensor(inputs).to(self.device)
        return inputs, labels

    def train(self, train_data, epochs=10, lr=0.001, batch_size=64):
        print('train...')
        self.net.train()
        op = optim.SGD(self.net.parameters(), lr, momentum=0.9)
        for epoch in range(epochs):
            loss_sum = 0
            iter_train = train_data.next(batch_size)
            for step, data in enumerate(iter_train, 0):
                inputs, labels = self.run_data(data)
                op.zero_grad()
                outputs = self.net.forward(inputs)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                op.step()

                loss_sum += loss.item()
                if step % 1000 == 999:
                    print('epoch: %2d, step: %4d loss: %.3f' %
                          (epoch + 1, step + 1, loss_sum / 1000))
                    loss_sum = 0

    def eval(self, eval_data):
        print('eval...')
        with torch.no_grad():
            self.net.eval()
            total = 0
            correct = 0
            iter_eval = eval_data.next()
            for step, data in enumerate(iter_eval, 0):
                inputs, labels = self.run_data(data)

                outputs = self.net.forward(inputs)
                _, targets = outputs.max(1)
                correct += (targets == labels).sum().item()
                total += labels.size()[0]
            print('total data: %d, acc: %.3f' % (total, correct / total))


lr = 0.01
epoch = 10
max_len = 80
batch_size = 64

print('load data...')
train_data = dataset('data/train.csv', max_len=max_len)
test_data = dataset('data/test.csv', max_len=max_len)
vocab_len = train_data.vocab_len

worker = Worker(vocab_len, max_len)
worker.train(train_data, epochs=epoch, lr=lr, batch_size=batch_size)
worker.eval(test_data)
