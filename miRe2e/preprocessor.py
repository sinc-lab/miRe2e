import torch as tr
import torch.nn as nn


class Preprocessor(nn.Module):
    def __init__(self, option, device="cpu"):
        super(Preprocessor, self).__init__()
        if option == 0:
            self.map = tr.Tensor([[0, 0, 0, 0],
                                  [1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1],
                                  [1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0],
                                  [0, 0, 0, 1]]).to(device)
        else:
            self.map = tr.Tensor([[0],
                                  [1],
                                  [-1]]).to(device)

    def forward(self, x):
        y = self.map[x, :].contiguous()
        return y
