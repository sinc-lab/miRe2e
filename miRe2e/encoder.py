from .resnet import *


class Encoder(nn.Module):
    def __init__(self, in_deep, width, n_resnets, n_blocks):
        super(Encoder, self).__init__()
        layers = []
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(in_deep))
        layers.append(nn.Conv1d(in_deep, width, kernel_size=3, padding=1))
        for j in range(n_blocks):
            for i in range(n_resnets):
                layers.append(ResNet(width))
            layers.append(nn.AvgPool1d(2))
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        return x


class EncoderStr(nn.Module):
    def __init__(self, in_deep, width, n_resnets, n_blocks):
        super(EncoderStr, self).__init__()
        layers = []
        layers.append(nn.ReLU())
        layers.append(nn.BatchNorm1d(in_deep))
        layers.append(nn.Conv1d(in_deep, width, kernel_size=3, padding=1))
        for j in range(n_blocks):
            for i in range(n_resnets):
                layers.append(ResNet(width))
            # layers.append(nn.AvgPool1d(2))
        self.conv_layers = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_layers(x)
        return x
