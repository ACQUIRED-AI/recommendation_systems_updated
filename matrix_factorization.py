import torch

class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors, user_map=None, item_map=None):
        super().__init__()
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.item_factors = torch.nn.Embedding(n_items, n_factors)

        torch.nn.init.normal_(self.user_factors.weight, std=0.05)
        torch.nn.init.normal_(self.item_factors.weight, std=0.05)

        # Save maps if provided
        self.user_map = user_map
        self.item_map = item_map

    def forward(self, user_idx, item_idx):
        return (self.user_factors(user_idx) * self.item_factors(item_idx)).sum(1)
