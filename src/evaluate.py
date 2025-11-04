import argparse
import glob
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset, Subset

from train import CompressionModel

plt.rcParams.update({"font.size": 15})


def get_test_dataset_from_checkpoint(checkpoint, data_dir):
    transform = transforms.Compose([transforms.ToTensor()])

    train_data = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )
    combined_dataset = ConcatDataset([train_data, test_data])

    test_indices = checkpoint["test_indices"]
    test_dataset = Subset(combined_dataset, test_indices)

    return test_dataset


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    checkpoint_paths = glob.glob(
        os.path.join(args.log_dir, "**", "model_best.pth.tar"), recursive=True
    )

    print(f"Found {len(checkpoint_paths)} models to evaluate.")

    rd_points = []
    reconstructions = []
    for checkpoint_path in checkpoint_paths:
        print(f"\n--- Evaluating model: {checkpoint_path} ---")

        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model_args = argparse.Namespace(**checkpoint["args"])

        model = CompressionModel(
            cae_feature_channels=model_args.cae_feature_channels,
            cae_latent_channels=model_args.cae_latent_channels,
            cae_res_blocks=model_args.cae_res_blocks,
            cm_feature_channels=model_args.cm_feature_channels,
            q_centers=model_args.q_centers,
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        test_dataset = get_test_dataset_from_checkpoint(checkpoint, model_args.data_dir)
        test_loader = DataLoader(
            test_dataset, batch_size=model_args.batch_size, shuffle=False
        )

        original_image, _ = test_dataset[1]
        original_image_batch = original_image.to(device).unsqueeze(0)
        original_image_cpu = original_image.cpu().squeeze()

        total_rate_symbol = 0.0
        total_distortion = 0.0
        mse_loss = nn.MSELoss()
        ce_loss = nn.CrossEntropyLoss(reduction="none")
        nats_to_bits = 1 / np.log(2)

        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                recon, logits, indices, mask = model(images)

                total_distortion += mse_loss(recon, images).item()

                flat_logits = logits.permute(0, 2, 3, 4, 1).reshape(
                    -1, model_args.q_centers
                )
                flat_indices = indices.view(-1)

                per_symbol_entropy_bits = (
                    ce_loss(flat_logits, flat_indices).view(indices.shape)
                    * nats_to_bits
                )

                masked_entropy_symbol = (per_symbol_entropy_bits * mask).mean()
                total_rate_symbol += masked_entropy_symbol.item()

        avg_rate_symbol = total_rate_symbol / len(test_loader)
        avg_distortion = total_distortion / len(test_loader)

        rd_points.append(
            {
                "h_target": model_args.h_target,
                "rate_symbol": avg_rate_symbol,
                "distortion": avg_distortion,
            }
        )
        print(f"    -> h_target: {model_args.h_target}")
        print(f"    -> Rate (bits/symbol): {avg_rate_symbol:.6f}")
        print(f"    -> Distortion (MSE): {avg_distortion:.6f}")

        with torch.no_grad():
            recon_single, _, _, _ = model(original_image_batch)

        recon_cpu = recon_single.cpu().squeeze(0).squeeze(0)

        reconstructions.append(
            {
                "recon": recon_cpu,
                "h_target": model_args.h_target,
                "original": original_image_cpu,
                "rate": avg_rate_symbol,
                "distortion": avg_distortion,
            }
        )

    output_dir = "figures"
    os.makedirs(output_dir, exist_ok=True)

    if len(rd_points) > 1:
        rd_points.sort(key=lambda p: p["rate_symbol"])

        all_distortions = [p["distortion"] for p in rd_points]
        all_rates = [p["rate_symbol"] for p in rd_points]

        plt.figure(figsize=(6, 6))
        plt.plot(
            all_rates,
            all_distortions,
            marker=None,
            linestyle="--",
            color="black",
            zorder=1,
        )
        unique_h_targets = sorted(list(set(p["h_target"] for p in rd_points)))
        markers = ["o", "s", "D", "^", "p", "v", "P", "*", "X"]
        for i, h_val in enumerate(unique_h_targets):
            rates_for_h = [
                p["rate_symbol"] for p in rd_points if p["h_target"] == h_val
            ]
            dists_for_h = [p["distortion"] for p in rd_points if p["h_target"] == h_val]
            plt.scatter(
                rates_for_h,
                dists_for_h,
                marker=markers[i % len(markers)],
                label=f"h_target = {h_val}",
                s=100,
                zorder=2,
            )
        plt.xlabel("Rate (average bits/symbol)")
        plt.ylabel("Distortion (MSE)")
        plt.title("Rate-Distortion Curve")
        plt.grid(True)
        plt.legend()
        save_path = os.path.join(output_dir, "rd_curve.png")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    reconstructions.sort(key=lambda p: p["h_target"])

    num_plots = len(reconstructions) + 1
    fig, axes = plt.subplots(1, num_plots, figsize=(num_plots * 3.5, 4.5))
    if num_plots == 2:
        axes = [axes[0], axes[1]]
    axes[0].imshow(reconstructions[0]["original"], cmap="gray")
    axes[0].set_title(f"Original")
    axes[0].axis("off")
    for i, data in enumerate(reconstructions):
        ax = axes[i + 1]
        ax.imshow(data["recon"], cmap="gray")
        title_str = f"Rec. (h_target = {data['h_target']})"
        ax.set_title(title_str)
        ax.axis("off")
    plt.tight_layout()
    recon_filename = os.path.join(output_dir, "reconstructions.png")
    plt.savefig(recon_filename, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=str)
    
    args = parser.parse_args()
    main(args)
