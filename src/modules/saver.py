import os
import torch


class Saver:
    """
    Handles saving and loading of model and optimizer states.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)

    def save_checkpoint(self, state, filename):
        filepath = os.path.join(self.save_dir, filename)
        torch.save(state, filepath)

    def save_best_model(self, state):
        best_filepath = os.path.join(self.save_dir, "model_best.pth.tar")
        torch.save(state, best_filepath)

    def load_checkpoint(
        self, checkpoint_path, model, optimizer_ae=None, optimizer_cm=None
    ):
        device = next(model.parameters()).device
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer_ae and "optimizer_ae_state_dict" in checkpoint:
            optimizer_ae.load_state_dict(checkpoint["optimizer_ae_state_dict"])

        if optimizer_cm and "optimizer_cm_state_dict" in checkpoint:
            optimizer_cm.load_state_dict(checkpoint["optimizer_cm_state_dict"])

        return checkpoint["epoch"]
