import torch.nn as nn


class FC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.act = nn.ReLU()

    def forward(self, x):
        y = self.fc1(x)
        y = self.act(y)
        y = self.fc2(y)
        y = self.act(y)
        y = self.fc3(y)
        return y
