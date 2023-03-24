import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class LRannealing(AbsWeighting):
    def __init__(self):
        super(LRannealing, self).__init__()

    def init_param(self):
        self.lambdas = torch.tensor([1.0]*(self.task_num-1), device=self.device)
        
    def backward(self, losses, **kwargs):
        alpha = kwargs['alpha']

        f_loss = losses[0]
        b_losses = losses[1:]

        grads = self._get_grads(losses, mode='backward', flatten=False)

        grads_f = grads[0]
        grads_bs = grads[1:]

        gradient_magnitudes = []
        for grad_f in grads_f:
            grad_magnitude = torch.norm(grad_f)
            gradient_magnitudes.append(grad_magnitude)
        max_grad_r = max(gradient_magnitudes)

        
        mean_grad_bs = []
        for i in range(self.task_num-1):
            gradient_magnitudes = []
            for grad_bs in grads_bs[i]:
                grad_magnitude = torch.norm(grad_bs)
                gradient_magnitudes.append(grad_magnitude)
            gradient_magnitudes = torch.stack(gradient_magnitudes)
            mean_grad_bs.append(torch.mean(gradient_magnitudes, dim=0))
        mean_grad_bs = torch.stack(mean_grad_bs)

        lambs_hat = max_grad_r / (mean_grad_bs + 1e-8)
        
        lambs = alpha*self.lambdas + (1-alpha)*lambs_hat
        self.lambdas = lambs
        
        lamb_b_loss_sum = torch.mul(lambs, b_losses)
        total_loss = f_loss + torch.sum(lamb_b_loss_sum)

        total_loss.backward()
        
        # update args
        updated_weights = torch.cat((torch.tensor([1.], device=self.device), self.lambdas)).detach().cpu().numpy()

        return updated_weights
