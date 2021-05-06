import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, C):
        super(ResNet, self).__init__()
        layers = []
        for i in range(2):
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(C))
            layers.append(nn.Conv1d(C, C, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) + x