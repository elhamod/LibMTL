import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class CW(AbsWeighting):
    r"""Constant Weighting (CW).

    """
    def __init__(self):
        super(CW, self).__init__()
    
    def init_param(self):
        self.weights = None

    def backward(self, losses, **kwargs):
        if self.weights is None:
            if kwargs['weights'] is None:
                self.weights = torch.tensor([1.0]*self.task_num)
            else:
                self.weights = torch.tensor( kwargs['weights'])
        loss = torch.mul(losses, self.weights.to(self.device)).sum()
        loss.backward()
        return np.ones(self.task_num)