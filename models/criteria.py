import torch
import torch.nn as nn

class MultiTaskUncertaintyLossKendall(nn.Module):

    def __init__(self, num_tasks):
        """
        Multi task loss with uncertainty weighting (Kendall 2016)
        \eta = 2*log(\sigma)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.eta = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        Input: 1-d tensor of scalar losses, shape (num_tasks,)
        Output: Total weighted loss
        """
        assert len(losses) == self.num_tasks, 'Expected {} losses to weight, got {}'.format(self.num_tasks, len(losses))
        # factor = torch.pow(2*torch.exp(self.eta) - 2, -1)
        factor = torch.exp(-self.eta)/2.
        total_loss = (losses * factor + self.eta).sum()
        return total_loss/self.num_tasks


class MultiTaskUncertaintyLossLiebel(nn.Module):

    def __init__(self, num_tasks):
        """
        Multi task loss with uncertainty weighting
        Liebel (2018) version
        Regularisation term ln(1 + sigma^2)
        """
        super().__init__()
        self.num_tasks = num_tasks
        self.sigma2 = nn.Parameter(0.25*torch.ones(num_tasks))

    def forward(self, losses):
        """
        Input: 1-d tensor of scalar losses, shape (num_tasks,)
        Output: Total weighted loss
        """
        assert len(losses) == self.num_tasks, 'Expected {} losses to weight, got {}'.format(self.num_tasks, len(losses))
        factor = 1./(2*self.sigma2)
        reg = torch.log(1. + self.sigma2) #regularisation term
        total_loss = (losses * factor + reg).sum()
        return total_loss/self.num_tasks

class DisturbLabelLoss(nn.Module):

    def __init__(self, device, disturb_prob=0.1):
        super(DisturbLabelLoss, self).__init__()
        self.disturb_prob = disturb_prob
        self.ce = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, pred, target):
        with torch.no_grad():
            disturb_indexes = torch.rand(len(pred)) < self.disturb_prob
            target[disturb_indexes] = torch.randint(pred.shape[-1], (int(disturb_indexes.sum()),)).to(self.device)
        return self.ce(pred, target)

    
class LabelSmoothingLoss(nn.Module):

    def __init__(self, smoothing=0.1, dim=-1):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.shape[-1] - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))


class TwoNeighbourSmoothingLoss(nn.Module):
    
    def __init__(self, smoothing=0.1, dim=-1):
        super().__init__()
        self.dim = dim
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        num_classes = pred.shape[self.dim]
        with torch.no_grad():
            targets = target.data.unsqueeze(1)
            true_dist = torch.zeros_like(pred)
            
            up_labels = targets.add(1)
            up_labels[up_labels >= num_classes] = num_classes - 2
            down_labels = targets.add(-1)
            down_labels[down_labels < 0] = 1
            
            smooth_values = torch.zeros_like(targets).float().add_(self.smoothing/2)
            true_dist.scatter_add_(1, up_labels, smooth_values)
            true_dist.scatter_add_(1, down_labels, smooth_values)
        
            true_dist.scatter_(1, targets, self.confidence)
            
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))