import torch
import torch.nn as nn


class Quantizer(nn.Module):
    """
    Quantizer.
    """

    def __init__(self, q_centers=16, sigma=1.0):
        super(Quantizer, self).__init__()
        self.sigma = sigma
        self.centers = nn.Parameter(torch.linspace(-1.0, 1.0, q_centers))

    def forward(self, z):
        z_expanded = z.unsqueeze(-1)
        centers_expanded = self.centers.view(1, 1, 1, -1)

        dist = (z_expanded - centers_expanded).pow(2)
        phi_soft = nn.functional.softmax(-self.sigma * dist, dim=-1)

        z_soft = torch.sum(phi_soft * self.centers, dim=-1)

        indices = dist.argmin(dim=-1)
        z_hard = self.centers[indices]

        z_bar = z_soft + (z_hard - z_soft).detach()

        return z_bar, indices
