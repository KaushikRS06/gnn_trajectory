"""
Training, loss, evaluation, and plotting utilities for trajectory models.

Provides ADE-based training, checkpoint selection, deterministic ADE/FDE
evaluation, and multimodal minADE/minFDE evaluation.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt


# For loss function
def ade_loss(predictions, targets):
    """Compute the Average Displacement Error (ADE) between predictions and targets."""
    errors = torch.norm(predictions - targets, dim=2)
    return errors.mean()  # Average over all timesteps and samples


# Training and validation loop
def train_and_validate(model, train_loader, val_loader, optimizer, criterion, num_epochs, checkpoint_path, device=None):
    """Train and validate a model for multiple epochs. Saves the parameters of the best model based on validation loss."""

    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")

    if device is None:
        device = next(model.parameters()).device

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')  # Initialize best validation loss to infinity

    # loop
    for epoch in range(num_epochs):

        # ==================
        # ==== TRAINING ====
        # ==================

        model.train()  # set training mode

        train_total_loss = 0.0
        train_examples = 0

        for batch in train_loader:

            batch = batch.to(device)

            optimizer.zero_grad()  # Reset gradients

            pred = model(batch)  # Forward pass

            target = batch.y.view(batch.num_graphs, 12, 2)  # Reshape target to match prediction shape

            loss = criterion(pred, target)  # Compute loss

            # Guard against non-finite loss values
            if not torch.isfinite(loss).item():
                raise ValueError("Non-finite training loss; training stopped")

            loss.backward()  # Backward pass
            optimizer.step()  # Update model parameters

            train_total_loss += (loss.item() * batch.num_graphs)  # Accumulate total loss
            train_examples += batch.num_graphs  # Accumulate number of examples

        # Compute average training loss for the epoch
        if train_examples == 0:
            raise ValueError("Training loader is empty")

        train_avg_loss = (train_total_loss / train_examples)
        train_losses.append(train_avg_loss)


        # ====================
        # ==== VALIDATION ====
        # ====================

        model.eval()  # set evaluation mode

        val_total_loss = 0.0
        val_examples = 0

        with torch.no_grad():  # Disable gradient computation for validation
            for batch in val_loader:

                batch = batch.to(device)

                pred = model(batch)  # Forward pass

                target = batch.y.view(batch.num_graphs, 12, 2)  # Reshape target

                loss = criterion(pred, target)  # Compute loss

                # Guard against non-finite loss values
                if not torch.isfinite(loss).item():
                    raise ValueError("Non-finite validation loss; training stopped")

                val_total_loss += (loss.item() * batch.num_graphs)  # Accumulate total loss
                val_examples += batch.num_graphs  # Accumulate number of examples

        # Compute average validation loss for the epoch
        if val_examples == 0:
            raise ValueError("Validation loader is empty")

        val_avg_loss = (val_total_loss / val_examples)
        val_losses.append(val_avg_loss)

        # Print
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Train ADE: {train_avg_loss:.4f} m | "
            f"Val ADE: {val_avg_loss:.4f} m"
        )

        # SAVE BEST CHECKPOINT
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            torch.save(
                model.state_dict(),
                checkpoint_path
            )

    return train_losses, val_losses

# ========================================================

def best_of_k_ade_loss(pred, target):
    """Mean minimum ADE across modes; pred has shape [B, K, 12, 2]."""
    if pred.ndim != 4 or pred.shape[1] < 1 or pred.shape[2:] != (12, 2):
        raise ValueError("Expected predictions with shape [B, K, 12, 2], K >= 1")
    if pred.shape[0] == 0 or target.numel() != pred.shape[0] * 12 * 2:
        raise ValueError("Expected one 12-step target per prediction sample")
    target = target.reshape(-1, 1, 12, 2)
    errors = torch.norm(pred - target, dim=-1)
    return errors.mean(dim=-1).min(dim=-1).values.mean()


def evaluate_multimodal_model(model, loader, device=None):
    """Return mean minADE and minFDE, minimized independently over modes.

    With the six-mode final model these are minADE6 and minFDE6.
    Unlike the historical notebook, FDE uses its own best mode rather
    than the mode selected by ADE.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    ade_list = []
    fde_list = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.y.reshape(batch.num_graphs, 1, 12, 2)
            errors = torch.norm(pred - target, dim=-1)
            ade_list.extend(errors.mean(dim=-1).min(dim=-1).values.cpu().numpy())
            fde_list.extend(errors[:, :, -1].min(dim=-1).values.cpu().numpy())

    if not ade_list:
        raise ValueError("Evaluation loader is empty")
    return np.mean(ade_list), np.mean(fde_list)


def evaluate_model(model, loader, device=None):
    """Compute mean ADE and FDE for a dataset."""

    if device is None:
        device = next(model.parameters()).device

    model.eval()

    ade_list = []
    fde_list = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            pred = model(batch) # Forward pass
            target = batch.y.view(batch.num_graphs, 12, 2)

            errors = torch.norm(pred - target, dim=2)  # Euclidean distance for each timestep

            # Compute ADE and FDE for each sample
            ade_per_sample = errors.mean(dim=1)
            fde_per_sample = errors[:, -1]

            ade_list.extend(ade_per_sample.cpu().numpy())
            fde_list.extend(fde_per_sample.cpu().numpy())

    if not ade_list:
        raise ValueError("Evaluation loader is empty")

    mean_ade = np.mean(ade_list)
    mean_fde = np.mean(fde_list)

    return mean_ade, mean_fde

# ========================================================

def plot_losses(train_losses, val_losses, title="Training and Validation Losses"):
    plt.figure(figsize=(10, 6))

    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)

    plt.legend()
    plt.grid()
    plt.show()
