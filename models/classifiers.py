import math
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

'''
AdaCos and Ad margin loss taken from https://github.com/4uiiurz1/pytorch-adacos
'''

class DropAffine(nn.Module):

    def __init__(self, num_features, num_classes):
        super(DropAffine, self).__init__()
        self.fc = nn.Linear(num_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()
        
    def forward(self, input, label=None):
        W = self.fc.weight
        b = self.fc.bias
        logits = F.linear(input, W, b)
        return logits


class GradientReversalFunction(Function):
    """
    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None


class GradientReversal(nn.Module):
    def __init__(self, lambda_=1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x, **kwargs):
        return GradientReversalFunction.apply(x, self.lambda_)



class L2SoftMax(nn.Module):

    def __init__(self, num_features, num_classes):
        super(L2SoftMax, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, input, label=None):
        x = F.normalize(input)
        W = F.normalize(self.W)
        logits = F.linear(x, W)
        return logits

class SoftMax(nn.Module):

    def __init__(self, num_features, num_classes):
        super(SoftMax, self).__init__()
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, input, label=None):
        x = input
        W = self.W
        logits = F.linear(x, W)
        return logits


class LinearUncertain(nn.Module):
    
    def __init__(self, in_features, out_features, bias=True):
        super(LinearUncertain, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias
        
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_beta = nn.Parameter(torch.Tensor(out_features, in_features))
        
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight_mu, gain=1.0)
        init_beta = np.log(np.exp(0.5 * np.sqrt(6)/(self.in_features+self.out_features)) - 1)
        nn.init.constant_(self.weight_beta, init_beta)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mu)
            bound = 1 / np.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    

    def forward(self, x):
        if self.training:
            eps = torch.randn(self.out_features, self.in_features).to(self.weight_mu.device)
            weights = self.weight_mu + torch.log(torch.exp(self.weight_beta) + 1) * eps
        else:
            weights = self.weight_mu
        return F.linear(x, weights, self.bias)


class XVecHead(nn.Module):

    def __init__(self, num_features, num_classes, hidden_features=None):
        super(XVecHead, self).__init__()
        hidden_features = num_features if not hidden_features else hidden_features
        self.fc_hidden = nn.Linear(num_features, hidden_features)
        self.nl = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(hidden_features)
        self.fc = nn.Linear(hidden_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, input, label=None, transform=False):
        input = self.fc_hidden(input)
        input = self.nl(input)
        input = self.bn(input)
        if transform:
            return input
        W = self.fc.weight
        b = self.fc.bias
        logits = F.linear(input, W, b)
        if logits.shape[-1] == 1:
            logits = torch.squeeze(logits, dim=-1)
        return logits

class XVecHeadUncertain(nn.Module):

    def __init__(self, num_features, num_classes, hidden_features=None):
        super().__init__()
        hidden_features = num_features if not hidden_features else hidden_features
        self.fc_hidden = LinearUncertain(num_features, hidden_features)
        self.nl = nn.LeakyReLU()
        self.bn = nn.BatchNorm1d(hidden_features)
        self.fc = LinearUncertain(hidden_features, num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, input, label=None, transform=False):
        input = self.fc_hidden(input)
        input = self.nl(input)
        input = self.bn(input)
        if transform:
            return input
        logits = self.fc(input)
        return logits


class AMSMLoss(nn.Module):

    def __init__(self, num_features, num_classes, s=30.0, m=0.4):
        super(AMSMLoss, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        target_logits = logits - self.m
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output



class SphereFace(nn.Module):

    def __init__(self, num_features, num_classes, s=30.0, m=1.35):
        super(SphereFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(self.m * theta)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output


class ArcFace(nn.Module):

    def __init__(self, num_features, num_classes, s=30.0, m=0.50):
        super(ArcFace, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = s
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # add margin
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        target_logits = torch.cos(theta + self.m)
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = logits * (1 - one_hot) + target_logits * one_hot
        # feature re-scale
        output *= self.s

        return output

class AdaCos(nn.Module):

    def __init__(self, num_features, num_classes, m=0.50):
        super(AdaCos, self).__init__()
        self.num_features = num_features
        self.n_classes = num_classes
        self.s = math.sqrt(2) * math.log(num_classes - 1)
        self.m = m
        self.W = nn.Parameter(torch.FloatTensor(num_classes, num_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)

    def forward(self, input, label=None):
        # normalize features
        x = F.normalize(input)
        # normalize weights
        W = F.normalize(self.W)
        # dot product
        logits = F.linear(x, W)
        if label is None:
            return logits
        # feature re-scale
        theta = torch.acos(torch.clamp(logits, -1.0 + 1e-7, 1.0 - 1e-7))
        one_hot = torch.zeros_like(logits)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        with torch.no_grad():
            B_avg = torch.where(one_hot < 1, torch.exp(self.s * logits), torch.zeros_like(logits))
            B_avg = torch.sum(B_avg) / input.size(0)
            theta_med = torch.median(theta[one_hot == 1])
            self.s = torch.log(B_avg) / torch.cos(torch.min(math.pi/4 * torch.ones_like(theta_med), theta_med))
        output = self.s * logits

        return output

