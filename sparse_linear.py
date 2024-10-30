import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist

import torch
import matplotlib.pyplot as plt
import math


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
        self.mask = nn.Parameter(self._create_initial_mask(), requires_grad=False)

        # Variable to store previous gradients
        self.previous_grads = None
        self.weight.register_hook(self._save_grad)

    def _save_grad(self, grad):
        """Hook function to save the gradient of the weight parameter."""
        self.previous_grads = grad

    @property
    def num_params(self):
        return self.in_features * self.out_features

    @property
    def target_num_sparse(self):
        num_params = self.in_features * self.out_features
        return int(num_params * self.sparse_fraction)
    
    @property
    def target_num_dense(self):
        return self.num_params - self.target_num_sparse

    def _create_initial_mask(self):
        """Initialize the mask based on the sparse_fraction."""
        num_params = self.num_params
        num_sparse = self.target_num_sparse
        mask = torch.ones(num_params, device=self.weight.device)
        mask[torch.randperm(num_params)[:num_sparse]] = 0
        return mask.view(self.out_features, self.in_features).bool()

    def _rigl_step(self):
        """Applies the RigL sparse-to-sparse training strategy without decay or scheduling."""

        if self.previous_grads is None:
            return
        
        device = self.weight.device
        # target_sparsity = self.sparse_fraction  # Fixed sparsity level
        target_num_dense = self.target_num_dense

        # Calculate scores for dropping and growing
        weight_magnitudes = torch.abs(self.weight).to(device)
        grad_magnitudes = torch.abs(self.previous_grads).to(device)

        print(grad_magnitudes.mean())
        
        exit()

        # Determine total connections and the target number of non-zero connections
        # total_params = self.weight.numel()
        # num_nonzero_target = int(total_params * (1 - target_sparsity))

        # Calculate number of elements to drop and grow to maintain exact sparsity
        num_current_nonzero = self.mask.sum().item()
        num_drop = max(num_current_nonzero - num_nonzero_target, 0)
        num_grow = num_drop

        # Drop Criterion: Drop the connections with the smallest weight magnitudes
        if num_drop > 0:
            _, drop_indices = torch.topk(-weight_magnitudes.view(-1), k=num_drop)
            new_mask = self.mask.view(-1).clone()
            new_mask[drop_indices] = 0

            # Grow Criterion: Grow connections with the largest gradient magnitudes
            grow_scores = torch.where(
                new_mask == 1, 
                torch.full_like(grad_magnitudes.view(-1), float('-inf'), device=device),  # Ensure device consistency
                grad_magnitudes.view(-1)
            )
            _, grow_indices = torch.topk(grow_scores, k=num_grow)
            new_mask[grow_indices] = 1

            # Reshape the mask and apply it to the layer
            self.mask = new_mask.view(self.out_features, self.in_features).to(device)

        assert self.mask.sum().item() == num_nonzero_target, f"Mask does not have the correct number of non-zero elements ({self.mask.sum().item()} vs {num_nonzero_target}), num_drop: {num_drop}, num_grow: {num_grow}"


    def forward(self, input):
        if self.training:
            self._rigl_step()

        # Apply mask to weights in forward pass
        masked_weight = self.weight * self.mask
        return nn.functional.linear(input, masked_weight)


def plot_sparse_linear_masks(model, max_plots: int = 16):
    """Plots the mask matrix of each SparseLinear layer in the model.
    """

    # Recursively find all SparseLinear layers
    sparse_layers = []
    for module in model.modules():
        if isinstance(module, SparseLinear):
            sparse_layers.append(module)
    sparse_layers = sparse_layers[:max_plots]  # Limit to max_plots

    # Calculate grid size
    num_layers = len(sparse_layers)
    n = math.ceil(math.sqrt(num_layers))
    
    # Create the figure and subplots
    fig, axes = plt.subplots(n, n, figsize=(12, 12))
    axes = axes.flatten()  # Flatten to easily iterate over axes

    # Plot each SparseLinear layer's mask matrix
    for i, layer in enumerate(sparse_layers):
        ax = axes[i]
        
        # Display the mask matrix
        mask_matrix = layer.mask.detach().cpu().numpy()
        ax.imshow(mask_matrix, cmap='gray', aspect='auto')
        
        # Calculate sparsity information
        total_elements = mask_matrix.size
        num_zeros = total_elements - mask_matrix.sum()
        sparsity = (num_zeros / total_elements) * 100
        
        # Set titles and axis labels
        ax.set_title(f"SL {i+1} - Sparsity: {sparsity:.2f}% ({num_zeros} zero-cells)")
        ax.set_xlabel(f"Input Features ({layer.in_features})")
        ax.set_ylabel(f"Output Features ({layer.out_features})")
        
        # Hide axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any unused subplots if num_layers is not a perfect square
    for j in range(i + 1, n * n):
        axes[j].axis("off")

    fig.tight_layout()

    return fig


if __name__ == "__main__":
    # Initialize a model with multiple SparseLinear layers
    model = nn.Sequential(
        SparseLinear(10, 8),
        nn.ReLU(),
        SparseLinear(8, 4),
        nn.ReLU(),
        SparseLinear(4, 2)
    )
    
    # Define a simple loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # Dummy input and target tensors
    input_tensor = torch.randn(2, 10)  # Batch size of 2, 10 input features
    target_tensor = torch.randn(2, 2)  # Batch size of 2, 2 output features

    for _ in range(10):
        # Forward pass
        output = model(input_tensor)
        
        # Calculate loss
        loss = criterion(output, target_tensor)
        print("Initial loss:", loss.item())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Perform optimization step
        optimizer.step()

    fig = plot_sparse_linear_masks(model)
    plt.show()