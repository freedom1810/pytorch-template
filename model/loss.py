import torch.nn.functional as F
import torch
import torch.nn as nn
# torch.nn.CrossEntropyLoss

def crossentropyloss(output, target):
    return torch.nn.CrossEntropyLoss(reduction='mean')(output, target)
     

class LabelSmoothingLoss(torch.nn.Module):
    def __init__(self, smoothing: float = 0.05, reduction="mean", weight=None, training=True):
        super(LabelSmoothingLoss, self).__init__()
        self.epsilon = smoothing
        self.reduction = reduction
        self.weight = weight
        self.training = training

    def reduce_loss(self, loss):
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss

    def linear_combination(self, x, y):
        #print(x, y, self.epsilon)
        return self.epsilon * x + (1 - self.epsilon) * y

    def forward(self, preds, target, TTA = False):
        target = target.to(dtype=torch.long)
        target = torch.squeeze(target)
        if self.weight is not None:
            self.weight = self.weight.to(preds.device)
        #print('---------------------')
        #print(target.shape, preds.shape)
        if self.training:
            #print('Train')
            n = preds.size(-1)
            log_preds = F.log_softmax(preds, dim=-1)
            loss = self.reduce_loss(-log_preds.sum(dim=-1))
            #print(log_preds.shape)
            nll = F.nll_loss(log_preds, target, reduction=self.reduction, weight=self.weight)
            return self.linear_combination(loss / n, nll)
        else:
            if not TTA:
                log_preds = F.log_softmax(preds, dim=-1)
            else:
                log_preds = preds
            return torch.nn.functional.cross_entropy(preds, target, weight=self.weight)

def labelsmoothingloss(output, target):
    return LabelSmoothingLoss(smoothing = 0.05)(output, target)



def nll_loss(output, target):
    return F.nll_loss(output, target)
def focal_binary_cross_entropy(logits, targets, gamma=2):
    num_label = targets.size(1)
    l = logits.reshape(-1)
    t = targets.reshape(-1)
    p = torch.sigmoid(l)
    p = torch.where(t >= 0.5, p, 1-p)
    logp = - torch.log(torch.clamp(p, 1e-4, 1-1e-4))
    loss = logp*((1-p)**gamma)
    loss = num_label*loss.mean()
    return loss
def bce_with_logits_loss(output, target):
    output = output.view(-1).float().cuda()
    target = target.view(-1).float().cuda()
    return F.binary_cross_entropy_with_logits(output, target)
