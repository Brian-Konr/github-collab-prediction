import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class TimeSeriesLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(TimeSeriesLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = F.relu(out)
        out = self.fc(out[:, -1, :])
        return out
