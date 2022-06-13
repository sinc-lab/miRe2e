import torch.nn as nn
import torch as tr


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True,
                 coef=tr.ones([2]), device="cpu"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

        if self.logits:
            self.BCE_loss = nn.BCEWithLogitsLoss(reduction='none',
                                                 pos_weight=coef.to(device))
        else:
            self.BCE_loss = nn.BCELoss(reduction='none')

    def forward(self, inputs, targets):

        loss = self.BCE_loss(inputs, targets)
        pt = tr.exp(-loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * loss

        if self.reduce:
            return tr.mean(F_loss)
        else:
            return F_loss
