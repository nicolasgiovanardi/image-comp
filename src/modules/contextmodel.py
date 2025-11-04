import torch
import torch.nn as nn


class MaskedConv3d(nn.Conv3d):
    """
    3D convolution with a causal mask.
    """

    def __init__(self, mask_type, *args, **kwargs):
        super(MaskedConv3d, self).__init__(*args, **kwargs)
        assert mask_type in ["A", "B"], "mask_type must be 'A' or 'B'"
        self.register_buffer("mask", torch.zeros_like(self.weight))

        _, _, D, H, W = self.weight.size()
        center_d, center_h, center_w = D // 2, H // 2, W // 2

        self.mask[:, :, :center_d, :, :] = 1
        self.mask[:, :, center_d, :center_h, :] = 1
        self.mask[:, :, center_d, center_h, :center_w] = 1

        if mask_type == "B":
            self.mask[:, :, center_d, center_h, center_w] = 1

    def forward(self, x):
        self.weight.data *= self.mask

        return super(MaskedConv3d, self).forward(x)


class ResidualBlock3d(nn.Module):
    """
    3D causal residual block using masked convolutions.
    """

    def __init__(self, num_channels):
        super(ResidualBlock3d, self).__init__()
        self.conv1 = MaskedConv3d(
            "B", num_channels, num_channels, kernel_size=3, padding=1
        )
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = MaskedConv3d(
            "B", num_channels, num_channels, kernel_size=3, padding=1
        )

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out)
        out += residual

        # return self.relu(out)
        return out


class ContextModel(nn.Module):
    """
    3D context model.
    """

    def __init__(self, cm_feature_channels, q_centers):
        super(ContextModel, self).__init__()
        self.model = nn.Sequential(
            MaskedConv3d("A", 1, cm_feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            ResidualBlock3d(cm_feature_channels),
            MaskedConv3d("B", cm_feature_channels, q_centers, kernel_size=3, padding=1),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        logits = self.model(x)

        return logits
