import torch
import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.net1 = nn.Linear(3072, 100)
        self.relu = nn.ReLU()
        self.net2 = nn.Linear(100, 10)
        self.softmax = nn.Softmax(dim=1)

        self.mean = torch.tensor([125.3, 123.0, 113.9]).to(device)
        self.std = torch.tensor([63.0, 62.1, 66.7]).to(device)

        self.to(device)

    def forward(self, x):
        x = self.preprocessing(x)
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        x = self.softmax(x)
        return x

    def preprocessing(self, x):
        x = x.reshape(-1, 3, 1024)
        x = (x-self.mean[None, :, None]) / self.std[None, :, None]
        x = x.reshape(-1, 3072)
        return x
