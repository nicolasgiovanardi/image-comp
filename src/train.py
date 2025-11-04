import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import ConcatDataset, DataLoader, Subset, random_split

from modules.autoencoder import CompressionAutoencoder
from modules.contextmodel import ContextModel
from modules.quantizer import Quantizer
from modules.saver import Saver
from modules.utils import create_log_dir, setup_logger


class CompressionModel(nn.Module):
    """
    Top-level model for data compression.
    """

    def __init__(
        self,
        cae_feature_channels,
        cae_latent_channels,
        cae_res_blocks,
        cm_feature_channels,
        q_centers,
    ):
        super(CompressionModel, self).__init__()
        self.cae_latent_channels = cae_latent_channels

        self.autoencoder = CompressionAutoencoder(
            cae_feature_channels, cae_latent_channels, cae_res_blocks
        )
        self.quantizer = Quantizer(q_centers)
        self.context_model = ContextModel(cm_feature_channels, q_centers)

    def get_importance_mask(self, latent_with_map):
        importance_map_channel = latent_with_map[:, 0:1, :, :]
        latent_z = latent_with_map[:, 1:, :, :]
        scaled_map = torch.sigmoid(importance_map_channel) * self.cae_latent_channels
        channel_indices = (
            torch.arange(self.cae_latent_channels, device=latent_with_map.device)
            .float()
            .view(1, -1, 1, 1)
        )
        mask = torch.maximum(
            torch.tensor(0.0),
            torch.minimum(torch.tensor(1.0), scaled_map - channel_indices + 1),
        )
        binarized_mask = (mask > 0.5).float()
        mask_bar = mask + (binarized_mask - mask).detach()

        return latent_z, mask_bar

    def forward(self, x):
        latent_with_map = self.autoencoder.forward_encoder(x)
        latent_z, mask = self.get_importance_mask(latent_with_map)
        latent_masked = latent_z * mask
        quantized_latent, indices = self.quantizer(latent_masked)
        logits = self.context_model(quantized_latent.detach())
        reconstruction = self.autoencoder.forward_decoder(quantized_latent)

        return reconstruction, logits, indices, mask


def main(args):
    if args.resume:
        log_dir = os.path.dirname(os.path.dirname(args.resume))
    else:
        log_dir = create_log_dir(args.log_dir_root)

    logger = setup_logger(log_dir)
    saver = Saver(save_dir=os.path.join(log_dir, "checkpoints"))
    recon_dir = os.path.join(log_dir, "reconstructions")
    os.makedirs(recon_dir, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.MNIST(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    combined_dataset = ConcatDataset([train_data, test_data])

    dataset_to_split = combined_dataset

    if args.total_subset_size is not None:
        if args.total_subset_size > len(combined_dataset):
            raise ValueError(
                "Subset size cannot be larger than the full 70,000 images."
            )

        np.random.seed(42)
        indices = np.random.choice(
            len(combined_dataset), args.total_subset_size, replace=False
        )
        dataset_to_split = Subset(combined_dataset, indices)

    dataset_size = len(dataset_to_split)
    val_split = 0.2
    test_split = 0.1

    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size - test_size

    generator = torch.Generator().manual_seed(42)
    train_subset, val_subset, test_subset = random_split(
        dataset_to_split, [train_size, val_size, test_size], generator=generator
    )

    train_dataset = Subset(dataset_to_split, train_subset.indices)
    val_dataset = Subset(dataset_to_split, val_subset.indices)

    if isinstance(dataset_to_split, Subset):
        train_indices = [dataset_to_split.indices[i] for i in train_subset.indices]
        val_indices = [dataset_to_split.indices[i] for i in val_subset.indices]
        test_indices = [dataset_to_split.indices[i] for i in test_subset.indices]
    else:
        train_indices = train_subset.indices
        val_indices = val_subset.indices
        test_indices = test_subset.indices

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4
    )

    logger.info(
        f"Dataset split: {train_size} training, {val_size} validation, {test_size} test samples."
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = CompressionModel(
        cae_feature_channels=args.cae_feature_channels,
        cae_latent_channels=args.cae_latent_channels,
        cae_res_blocks=args.cae_res_blocks,
        cm_feature_channels=args.cm_feature_channels,
        q_centers=args.q_centers,
    ).to(device)

    optimizer_ae = optim.Adam(
        list(model.autoencoder.parameters()) + list(model.quantizer.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer_cm = optim.Adam(
        model.context_model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scheduler_ae = optim.lr_scheduler.StepLR(
        optimizer_ae, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )
    scheduler_cm = optim.lr_scheduler.StepLR(
        optimizer_cm, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )

    start_epoch = 0
    if args.resume:
        start_epoch = saver.load_checkpoint(
            args.resume, model, optimizer_ae, optimizer_cm
        )

    best_loss = float("inf")
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss(reduction="none")

    nats_to_bits = 1 / np.log(2)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_rd_loss, train_cm_loss, train_distortion, train_rate = 0.0, 0.0, 0.0, 0.0
        for i, (images, _) in enumerate(train_loader):
            images = images.to(device)
            recon, logits, indices, mask = model(images)

            distortion_loss = mse_loss(recon, images)
            flat_logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, args.q_centers)
            flat_indices = indices.view(-1)
            per_symbol_entropy = (
                ce_loss(flat_logits, flat_indices).view(indices.shape) * nats_to_bits
            )
            context_model_loss = per_symbol_entropy.mean()
            masked_entropy = (per_symbol_entropy * mask).mean()
            clipped_rate_term = args.beta * torch.relu(masked_entropy - args.h_target)
            rate_distortion_loss = distortion_loss + clipped_rate_term

            optimizer_ae.zero_grad()
            rate_distortion_loss.backward(retain_graph=True)
            optimizer_ae.step()

            optimizer_cm.zero_grad()
            context_model_loss.backward()
            optimizer_cm.step()

            train_rd_loss += rate_distortion_loss.item()
            train_distortion += distortion_loss.item()
            train_rate += masked_entropy.item()
            train_cm_loss += context_model_loss.item()

        avg_train_rd_loss = train_rd_loss / len(train_loader)
        avg_train_distortion = train_distortion / len(train_loader)
        avg_train_rate = train_rate / len(train_loader)
        avg_train_cm_loss = train_cm_loss / len(train_loader)

        model.eval()
        val_rd_loss, val_cm_loss, val_distortion, val_rate = 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for i, (images, _) in enumerate(val_loader):
                images = images.to(device)
                recon, logits, indices, mask = model(images)

                distortion_loss = mse_loss(recon, images)
                flat_logits = logits.permute(0, 2, 3, 4, 1).reshape(-1, args.q_centers)
                flat_indices = indices.view(-1)
                per_symbol_entropy = (
                    ce_loss(flat_logits, flat_indices).view(indices.shape)
                    * nats_to_bits
                )
                context_model_loss = per_symbol_entropy.mean()
                masked_entropy = (per_symbol_entropy * mask).mean()
                clipped_rate_loss = args.beta * torch.relu(
                    masked_entropy - args.h_target
                )
                rate_distortion_loss = distortion_loss + clipped_rate_loss

                val_rd_loss += rate_distortion_loss.item()
                val_distortion += distortion_loss.item()
                val_rate += masked_entropy.item()
                val_cm_loss += context_model_loss.item()

                if i == 0:
                    val_batch_size = images.size(0)
                    comparison = torch.cat(
                        [images[:8], recon.view(val_batch_size, 1, 28, 28)[:8]]
                    )
                    save_image(
                        comparison.cpu(),
                        os.path.join(recon_dir, f"recon_epoch_{epoch+1}.png"),
                        nrow=8,
                    )

        scheduler_ae.step()
        scheduler_cm.step()

        avg_val_rd_loss = val_rd_loss / len(val_loader)
        avg_val_distortion = val_distortion / len(val_loader)
        avg_val_rate = val_rate / len(val_loader)
        avg_val_cm_loss = val_cm_loss / len(val_loader)

        log_message = (
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Train RD Loss: {avg_train_rd_loss:.6f} | "
            f"Train P Loss: {avg_train_cm_loss:.6f} | "
            f"Train R: {avg_train_rate:.6f} | "
            f"Train D: {avg_train_distortion:.6f} | "
            f"Val RD Loss: {avg_val_rd_loss:.6f} | "
            f"Val P Loss: {avg_val_cm_loss:.6f} | "
            f"Val R: {avg_val_rate:.6f} | "
            f"Val D: {avg_val_distortion:.6f}"
        )
        logger.info(log_message)

        is_best = avg_val_rd_loss < best_loss
        if is_best:
            best_loss = avg_val_rd_loss
            logger.info(
                f">> New best validation loss: {best_loss:.6f}. Saving best model..."
            )
            saver.save_best_model(
                {
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "loss": avg_val_rd_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_ae_state_dict": optimizer_ae.state_dict(),
                    "optimizer_cm_state_dict": optimizer_cm.state_dict(),
                    "train_indices": train_indices,
                    "validation_indices": val_indices,
                    "test_indices": test_indices,
                }
            )

        if (epoch + 1) % args.save_interval == 0:
            logger.info(f">> Saving periodic checkpoint for epoch {epoch+1}...")
            saver.save_checkpoint(
                {
                    "args": vars(args),
                    "epoch": epoch + 1,
                    "loss": avg_val_rd_loss,
                    "model_state_dict": model.state_dict(),
                    "optimizer_ae_state_dict": optimizer_ae.state_dict(),
                    "optimizer_cm_state_dict": optimizer_cm.state_dict(),
                    "train_indices": train_indices,
                    "validation_indices": val_indices,
                    "test_indices": test_indices,
                },
                filename=f"checkpoint_epoch_{epoch+1}.pth.tar",
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--log_dir_root", type=str, default="logs")
    parser.add_argument(
        "--total_subset_size",
        type=int,
        default=None,
    )

    parser.add_argument("--cae_feature_channels", type=int, default=128)
    parser.add_argument("--cae_latent_channels", type=int, default=64)
    parser.add_argument("--cae_res_blocks", type=int, default=6)
    parser.add_argument("--cm_feature_channels", type=int, default=32)
    parser.add_argument("--q_centers", type=int, default=16)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay_step", type=int, default=50)
    parser.add_argument("--lr_decay_gamma", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--h_target", type=float, default=1.0)

    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None)

    args = parser.parse_args()
    main(args)
