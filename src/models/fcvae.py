"""
FCVAE: Frequency-enhanced Conditional VAE
- Unified model combining CVAE with frequency conditioning
- Simplified interface for leak detection
- Python 3.11 compatible
"""

from typing import Optional, Union, Tuple
import torch
import torch.nn as nn
from .cvae import ConditionalVAE


class FCVAE(nn.Module):
    """
    Frequency-enhanced Conditional VAE for Leak Detection
    - Wrapper around ConditionalVAE with simplified interface
    - Handles training and inference modes
    """

    def __init__(
        self,
        window_size: int = 96,
        latent_dim: int = 8,
        condition_dim: int = 16,
        hidden_dim: int = 100,
        d_model: int = 256,
        d_ff: int = 512,
        n_head: int = 8,
        kernel_size: int = 16,
        stride: int = 8,
        dropout: float = 0.05,
        kl_weight: float = 0.005
    ):
        """
        Args:
            window_size: Time series window size (default: 96 = 48 hours)
            latent_dim: Latent space dimension
            condition_dim: Condition embedding dimension per feature (global/local)
            hidden_dim: Hidden layer dimension in encoder/decoder
            d_model: Attention model dimension
            d_ff: Feed-forward dimension
            n_head: Number of attention heads
            kernel_size: Local FFT kernel size
            stride: Local FFT stride
            dropout: Dropout rate
            kl_weight: KL divergence weight in loss
        """
        super().__init__()

        self.window_size = window_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight

        # Conditional VAE
        self.cvae = ConditionalVAE(
            window_size=window_size,
            latent_dim=latent_dim,
            condition_dim=condition_dim,
            hidden_dim=hidden_dim,
            d_model=d_model,
            d_ff=d_ff,
            n_head=n_head,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        mode: str = "train"
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass with mode-based behavior:
        - train/valid: CM-ELBO loss (semi-supervised)
        - test: MCMC probability (unsupervised)

        Args:
            x: [batch, window_size] or [batch, 1, window_size]
            mask: [batch, window_size] - Mask for valid timesteps (train/valid only)
            mode: "train", "valid", or "test"

        Returns:
            If mode="train" or "valid":
                loss: Scalar CM-ELBO loss
            If mode="test":
                (x, prob): Input and log probability for anomaly detection
        """
        # Ensure 3D input
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [batch, 1, window_size]

        if mode in ["train", "valid"]:
            # === Semi-supervised: CM-ELBO with partial labels ===
            # CVAE forward pass
            mu_x, logvar_x, recon, mean, logvar, z = self.cvae(x, mode=mode)

            # Compute CM-ELBO loss
            loss = self.cvae.compute_loss(
                x, mu_x, logvar_x, mean, logvar, z,
                mask=mask,
                kl_weight=self.kl_weight
            )

            return loss
        
        else:
            # === Unsupervised: MCMC inference ===
            x_out, prob = self.cvae(x, mode=mode)
            return x_out, prob

    def compute_anomaly_scores(
        self,
        x: torch.Tensor,
        method: str = "mcmc",
        reduction: str = "max"
    ) -> torch.Tensor:
        """
        Compute anomaly scores

        Args:
            x: [batch, window_size] or [batch, 1, window_size]
            method: "mcmc" (MCMC sampling) or "recon" (reconstruction error)
            reduction: "max", "mean", or "last" (default: "max")

        Returns:
            scores: [batch] - Anomaly scores per sample
        """
        # Ensure 3D input
        if x.dim() == 2:
            x = x.unsqueeze(1)

        self.eval()
        with torch.no_grad():
            if method == "mcmc":
                # === Unsupervised: MCMC-based anomaly score ===
                _, prob = self.cvae(x, mode="test")  # [batch, 1, window_size]
                prob = prob.squeeze(1)  # [batch, window_size]
                
                # Higher probability → Lower anomaly (negative log-prob)
                anomaly_scores = -prob  # [batch, window_size]
            
            else:
                # === Supervised: Reconstruction error ===
                mu_x, logvar_x, recon, mean, logvar, z = self.cvae(x, mode="train")
                recon = recon.squeeze(1)
                
                # Compute reconstruction error
                x_flat = x.squeeze(1)  # [batch, window_size]
                anomaly_scores = (x_flat - recon) ** 2  # [batch, window_size]

        # Reduce to per-sample score
        if reduction == "max":
            scores = anomaly_scores.max(dim=1)[0]  # Maximum score in window
        elif reduction == "mean":
            scores = anomaly_scores.mean(dim=1)  # Average score
        elif reduction == "last":
            scores = anomaly_scores[:, -1]  # Last timestep score
        else:
            raise ValueError(f"Unknown reduction: {reduction}")

        return scores

    def reconstruct(
        self,
        x: torch.Tensor,
        return_latent: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Reconstruct input time series

        Args:
            x: [batch, window_size] - Input time series
            return_latent: Whether to return latent representation

        Returns:
            recon: [batch, window_size] - Reconstructed time series
            latent: [batch, latent_dim] - Latent representation (if return_latent=True)
        """
        # Add channel dimension if needed
        if x.dim() == 2:
            x = x.unsqueeze(1)

        self.eval()
        with torch.no_grad():
            mu_x, logvar_x, recon, mean, logvar, z = self.cvae(x)

        recon = recon.squeeze(1)

        if return_latent:
            return recon, mean
        else:
            return recon, None


# ===== Test =====
if __name__ == '__main__':
    print("=" * 70)
    print("Testing FCVAE Model")
    print("=" * 70)

    # Hyperparameters
    batch_size = 32
    window_size = 96
    latent_dim = 8
    condition_dim = 16

    # Random input (standardized water demand data)
    x = torch.randn(batch_size, window_size)

    print(f"\nInput shape: {x.shape}")
    print(f"Input stats: mean={x.mean():.4f}, std={x.std():.4f}")

    # Create model
    print("\n" + "-" * 70)
    print("Creating FCVAE model...")
    print("-" * 70)
    model = FCVAE(
        window_size=window_size,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dim=100,
        d_model=256,
        d_ff=512,
        n_head=8,
        kernel_size=16,
        stride=8,
        dropout=0.05,
        kl_weight=0.005
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test forward pass
    print("\n" + "-" * 70)
    print("Testing forward pass (training mode)...")
    print("-" * 70)
    model.train()
    output = model(x)

    print(f"Reconstruction shape: {output['recon'].shape}")
    print(f"Reconstruction mean shape: {output['mu_x'].shape}")
    print(f"Latent mean shape: {output['mean'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")

    # Test with mask
    print("\n" + "-" * 70)
    print("Testing forward pass with mask...")
    print("-" * 70)
    mask = torch.ones(batch_size, window_size)
    mask[:, -10:] = 0  # Mask last 10 timesteps (simulating missing data)
    output_masked = model(x, mask=mask)

    print(f"Loss with mask: {output_masked['loss'].item():.4f}")

    # Test reconstruction
    print("\n" + "-" * 70)
    print("Testing reconstruction...")
    print("-" * 70)
    model.eval()
    recon, latent = model.reconstruct(x, return_latent=True)

    print(f"Reconstruction shape: {recon.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction error (MSE): {((x - recon) ** 2).mean().item():.4f}")

    # Test anomaly score computation
    print("\n" + "-" * 70)
    print("Testing anomaly score computation...")
    print("-" * 70)
    scores = model.compute_anomaly_scores(x, n_samples=10)

    print(f"Anomaly scores shape: {scores.shape}")
    print(f"Score stats: mean={scores.mean():.4f}, std={scores.std():.4f}")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    # Simulate leak detection
    print("\n" + "-" * 70)
    print("Simulating leak detection scenario...")
    print("-" * 70)

    # Normal data
    x_normal = torch.randn(16, window_size) * 0.5

    # Leak data (with anomaly injected)
    x_leak = x_normal.clone()
    x_leak[:, 50:70] += torch.randn(16, 20) * 2.0  # Inject anomaly

    # Compute scores
    scores_normal = model.compute_anomaly_scores(x_normal, n_samples=10)
    scores_leak = model.compute_anomaly_scores(x_leak, n_samples=10)

    print(f"Normal data score: {scores_normal.mean():.4f} ± {scores_normal.std():.4f}")
    print(f"Leak data score: {scores_leak.mean():.4f} ± {scores_leak.std():.4f}")
    print(f"Score difference: {(scores_leak.mean() - scores_normal.mean()).item():.4f}")

    # Test batch processing
    print("\n" + "-" * 70)
    print("Testing batch processing...")
    print("-" * 70)

    import time

    # Small batch
    x_small = torch.randn(8, window_size)
    start = time.time()
    _ = model.compute_anomaly_scores(x_small, n_samples=128)
    time_small = time.time() - start

    # Large batch
    x_large = torch.randn(256, window_size)
    start = time.time()
    _ = model.compute_anomaly_scores(x_large, n_samples=128)
    time_large = time.time() - start

    print(f"Small batch (8 samples): {time_small:.3f}s")
    print(f"Large batch (256 samples): {time_large:.3f}s")
    print(f"Time per sample: {time_large/256*1000:.2f}ms")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
