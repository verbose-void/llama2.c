import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

class SparseLinear(nn.Module):
    def __init__(self, in_features, out_features, sparse_fraction=0.9, alpha=0.3, bias: bool = False):
        """A sparse version of the standard Linear layer. Implements RigL's sparse-to-sparse strategy.

        Args:
            in_features (int): Number of input features.
            out_features (int): Number of output features.
            sparse_fraction (float): Percentage of weights that will be masked out to zero.
            alpha (float): Fraction of weights to drop/grow at each step.
            bias (bool): Whether to include a bias term in the linear transformation.
        """

        super(SparseLinear, self).__init__()

        if bias: 
            raise NotImplementedError("SparseLinear does not support bias yet")
        self.bias = None
        
        self.in_features = in_features
        self.out_features = out_features
        self.sparse_fraction = sparse_fraction
        self.alpha = alpha

        # Weight initialization
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))

        # Initial mask
        self.mask = self._create_initial_mask()
        
        # Gradient storage for RigL updates
        self.dense_grad = torch.zeros_like(self.weight)

    def _create_initial_mask(self):
        """Initialize the mask based on the sparse_fraction."""
        num_params = self.in_features * self.out_features
        num_sparse = int(num_params * self.sparse_fraction)
        mask = torch.ones(num_params, device=self.weight.device)
        mask[torch.randperm(num_params)[:num_sparse]] = 0
        return mask.view(self.out_features, self.in_features).bool()

    def _rigl_step(self):
        """Updates the mask by applying RigL's sparse-to-sparse strategy."""
        device = self.weight.device  # Ensure all tensors are on the same device
        drop_fraction = self.alpha / 2

        # Ensure self.mask is on the same device as self.weight
        self.mask = self.mask.to(device)

        # Calculate drop/grow scores
        score_drop = torch.abs(self.weight).to(device)
        score_grow = torch.abs(self.dense_grad).to(device)

        # Handle distributed environment
        if dist.is_initialized():
            world_size = dist.get_world_size()
            dist.all_reduce(score_drop)
            dist.all_reduce(score_grow)
            score_drop /= world_size
            score_grow /= world_size

        # Determine drop and grow counts
        total_params = self.weight.numel()
        num_nonzero = self.mask.sum().item()
        num_drop = int(num_nonzero * drop_fraction)
        num_grow = num_drop

        # Create drop mask
        _, drop_indices = torch.topk(score_drop.view(-1), k=total_params)
        new_mask = torch.ones_like(score_drop.view(-1), dtype=torch.bool, device=device)
        new_mask[drop_indices[:num_drop]] = 0

        # Grow new connections based on grow scores
        grow_score = torch.where(
            self.mask.view(-1), 
            torch.full_like(score_grow.view(-1), float('-inf'), device=device), 
            score_grow.view(-1)
        )
        _, grow_indices = torch.topk(grow_score, k=num_grow)
        new_mask[grow_indices] = 1
        self.mask = new_mask.view(self.out_features, self.in_features).to(device)

        # Reset dense_grad after each update
        self.dense_grad.zero_()


    def forward(self, input):
        # Apply RigL step to update the mask every forward pass
        self._rigl_step()

        # Apply mask to weights in forward pass
        masked_weight = self.weight * self.mask
        return nn.functional.linear(input, masked_weight)

    def backward(self, grad_output):
        # Mask the gradients in the backward pass
        masked_grad = grad_output * self.mask
        return masked_grad