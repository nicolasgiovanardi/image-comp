import torch.nn as nn


class ResidualBlock2d(nn.Module):
    """
    2D residual block.
    """

    def __init__(self, num_channels):
        super(ResidualBlock2d, self).__init__()
        self.conv1 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            num_channels, num_channels, kernel_size=3, stride=1, padding=1
        )
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual

        # return self.relu(out)
        return out


class CompressionAutoencoder(nn.Module):
    """
    2D autoencoder model for MNIST.
    """

    def __init__(self, cae_feature_channels, cae_latent_channels, cae_res_blocks):
        super(CompressionAutoencoder, self).__init__()

        self.encoder_res_blocks = nn.ModuleList(
            [ResidualBlock2d(cae_feature_channels) for _ in range(cae_res_blocks)]
        )
        self.decoder_res_blocks = nn.ModuleList(
            [ResidualBlock2d(cae_feature_channels) for _ in range(cae_res_blocks)]
        )

        self.final_encoder_res_block = ResidualBlock2d(cae_feature_channels)
        self.final_decoder_res_block = ResidualBlock2d(cae_feature_channels)

        self.final_encoder_conv = nn.Conv2d(
            cae_feature_channels, cae_latent_channels + 1, kernel_size=3, padding=1
        )
        self.initial_decoder_conv = nn.Conv2d(
            cae_latent_channels, cae_feature_channels, kernel_size=3, padding=1
        )

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, cae_feature_channels // 2, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(cae_feature_channels // 2),
            nn.ReLU(True),
            nn.Conv2d(
                cae_feature_channels // 2,
                cae_feature_channels,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.BatchNorm2d(cae_feature_channels),
            nn.ReLU(True),
        )
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(
                cae_feature_channels,
                cae_feature_channels // 2,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.BatchNorm2d(cae_feature_channels // 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                cae_feature_channels // 2,
                1,
                kernel_size=5,
                stride=2,
                padding=2,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

    def forward_encoder(self, x):
        net = self.encoder_conv(x)
        long_residual_input = net

        num_blocks = len(self.encoder_res_blocks)
        for i in range(0, num_blocks, 3):
            short_residual_input = net
            net = self.encoder_res_blocks[i](net)
            if (i + 1) < num_blocks:
                net = self.encoder_res_blocks[i + 1](net)
            if (i + 2) < num_blocks:
                net = self.encoder_res_blocks[i + 2](net)
            net = net + short_residual_input

        net = self.final_encoder_res_block(net)
        net = net + long_residual_input
        net = self.final_encoder_conv(net)

        return net

    def forward_decoder(self, q):
        net = self.initial_decoder_conv(q)
        long_residual_input = net

        num_blocks = len(self.decoder_res_blocks)
        for i in range(0, num_blocks, 3):
            short_residual_input = net
            net = self.decoder_res_blocks[i](net)
            if (i + 1) < num_blocks:
                net = self.decoder_res_blocks[i + 1](net)
            if (i + 2) < num_blocks:
                net = self.decoder_res_blocks[i + 2](net)
            net = net + short_residual_input

        net = self.final_decoder_res_block(net)
        net = net + long_residual_input
        net = self.decoder_deconv(net)

        return net
