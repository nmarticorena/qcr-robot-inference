import torch
import wandb


def rs_imle_loss(real_samples, fake_samples, epsilon=0.03, train_step: int | None = None):
    B, T, D = real_samples.shape
    n_samples = fake_samples.shape[1]

    real_flat = real_samples.reshape(B, 1, -1)
    fake_flat = fake_samples.reshape(B, n_samples, -1)

    distances = torch.cdist(real_flat, fake_flat).squeeze(1)
    valid_samples = (distances > epsilon).float()
    log_kwargs = {"step": train_step} if train_step is not None else {}
    wandb.log(
        {
            "max_distance": distances.max().item(),
            "min_distance": distances.min().item(),
            "mean_distance": distances.mean().item(),
            "epsilon": epsilon,
        },
        **log_kwargs,
    )
    min_distances, _ = (distances + (1 - valid_samples) * distances.max()).min(dim=1)
    valid_real_samples = (min_distances < distances.max()).float()
    if valid_real_samples.sum() > 0:
        loss = (min_distances * valid_real_samples).sum() / valid_real_samples.sum()
    else:
        loss = torch.tensor(0.0, device=real_samples.device)
    return loss
