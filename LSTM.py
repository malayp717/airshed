import torch
import torch.nn as nn

class LSTM(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, device):
        super(LSTM, self).__init__()
        self.device = device

        self.lstm = nn.LSTM(input_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, out_dim)

    def forward(self, X):
        X = X.to(self.device)

        out, (h, _) = self.lstm(X)
        out = self.fc(out)

        return out, h