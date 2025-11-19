"""
Conditional Variational Autoencoder (CVAE)
- Frequency-conditioned VAE for time series anomaly detection
- Learns normal patterns and reconstructs input
- Python 3.11 compatible
"""

from typing import Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import SelfAttentionLayer


class FrequencyConditionExtractor(nn.Module):
    """
    Frequency-based Condition Extractor
    - Extracts global and local frequency features using FFT
    - Combines with self-attention for rich representations
    """

    def __init__(
        self,
        window_size: int,
        condition_dim: int,
        d_model: int = 256,
        d_ff: int = 512,
        n_head: int = 8,
        kernel_size: int = 16,
        stride: int = 8,
        dropout: float = 0.1
    ):
        """
        Args:
            window_size: Time series window size
            condition_dim: Output condition embedding dimension
            d_model: Model dimension for attention
            d_ff: Feed-forward dimension
            n_head: Number of attention heads
            kernel_size: Local FFT kernel size
            stride: Stride for sliding window
            dropout: Dropout rate
        """
        super().__init__()

        self.window_size = window_size
        self.kernel_size = kernel_size
        self.stride = stride

        # Global frequency embedding
        # rfft output size: (window_size + 2) // 2 for real + imag parts
        global_fft_size = ((window_size - 1) // 2 + 1) * 2  # real + imag concatenated
        self.emb_global = nn.Sequential(
            nn.Linear(global_fft_size, condition_dim),
            nn.Tanh(),
        )

        # Local frequency embedding
        # rfft output size for kernel: (kernel_size // 2 + 1) * 2 (real + imag)
        local_fft_size = (kernel_size // 2 + 1) * 2
        self.emb_local = nn.Sequential(
            nn.Linear(local_fft_size, d_model),
            nn.Tanh(),
        )

        # Self-attention for local features
        self.attention = SelfAttentionLayer(
            d_model=d_model,
            d_ff=d_ff,
            n_head=n_head,
            d_k=d_model // n_head,
            d_v=d_model // n_head,
            dropout=dropout
        )

        # Output projection
        self.out_linear = nn.Sequential(
            nn.Linear(d_model, condition_dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract frequency conditions from input

        Args:
            x: [batch, 1, window_size]

        Returns:
            condition: [batch, 1, 2 * condition_dim] (global + local)
        """
        batch_size = x.size(0)

        # === Global Frequency Features ===
        # FFT of entire sequence (exclude last point for stability)
        x_g = x[:, :, :-1]  # [batch, 1, window_size-1]
        f_global = torch.fft.rfft(x_g, dim=-1)  # Complex FFT
        f_global = torch.cat([f_global.real, f_global.imag], dim=-1)  # [batch, 1, freq_bins]
        f_global = self.emb_global(f_global)  # [batch, 1, condition_dim]

        # === Local Frequency Features ===
        # Prepare for sliding window: [batch, 1, 1, window_size]
        x_l = x.view(batch_size, 1, 1, -1)
        x_l[:, :, :, -1] = 0  # Zero out last point

        # Sliding window using Unfold
        unfold = nn.Unfold(
            kernel_size=(1, self.kernel_size),
            stride=(1, self.stride),
            padding=0
        )
        unfold_x = unfold(x_l)  # [batch, kernel_size, num_windows]
        unfold_x = unfold_x.transpose(1, 2)  # [batch, num_windows, kernel_size]

        # FFT on each local window
        f_local = torch.fft.rfft(unfold_x, dim=-1)
        f_local = torch.cat([f_local.real, f_local.imag], dim=-1)  # [batch, num_windows, 2+kernel_size]

        # Embed local features
        f_local = self.emb_local(f_local)  # [batch, num_windows, d_model]

        # Apply self-attention across windows
        f_local, _ = self.attention(f_local)  # [batch, num_windows, d_model]

        # Take the last window's representation
        f_local = f_local[:, -1, :].unsqueeze(1)  # [batch, 1, d_model]
        f_local = self.out_linear(f_local)  # [batch, 1, condition_dim]

        # Concatenate global and local features
        condition = torch.cat([f_global, f_local], dim=-1)  # [batch, 1, 2*condition_dim]

        return condition


class ConditionalVAE(nn.Module):
    """
    Conditional Variational Autoencoder
    - Encoder: Input + Condition → Latent space (mean, logvar)
    - Decoder: Latent + Condition → Reconstruction (mean, logvar)
    """

    def __init__(
        self,
        window_size: int,
        latent_dim: int = 8,
        condition_dim: int = 16,
        hidden_dim: int = 100,
        d_model: int = 256,
        d_ff: int = 512,
        n_head: int = 8,
        kernel_size: int = 16,
        stride: int = 8,
        dropout: float = 0.05
    ):
        """
        Args:
            window_size: Time series window size
            latent_dim: Latent space dimension
            condition_dim: Condition embedding dimension (per global/local)
            hidden_dim: Hidden layer dimension
            d_model: Attention model dimension
            d_ff: Feed-forward dimension
            n_head: Number of attention heads
            kernel_size: Local FFT kernel size
            stride: Local FFT stride
            dropout: Dropout rate
        """
        super().__init__()

        self.window_size = window_size
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim

        # Frequency condition extractor
        self.condition_extractor = FrequencyConditionExtractor(
            window_size=window_size,
            condition_dim=condition_dim,
            d_model=d_model,
            d_ff=d_ff,
            n_head=n_head,
            kernel_size=kernel_size,
            stride=stride,
            dropout=dropout
        )

        # === Encoder ===
        # Input: [window_size + 2*condition_dim]
        encoder_input_dim = window_size + 2 * condition_dim

        self.encoder = nn.Sequential(
            nn.Linear(encoder_input_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout)
        )

        # Latent space parameters
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Sequential(
            nn.Linear(hidden_dim, latent_dim),
            nn.Softplus()  # Ensure positive variance
        )

        # === Decoder ===
        # Input: [latent_dim + 2*condition_dim]
        decoder_input_dim = latent_dim + 2 * condition_dim

        self.decoder_input = nn.Linear(decoder_input_dim, hidden_dim)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, window_size),
            nn.Tanh()
        )

        # Reconstruction distribution parameters
        self.fc_mu_x = nn.Linear(window_size, window_size)
        self.fc_logvar_x = nn.Sequential(
            nn.Linear(window_size, window_size),
            nn.Softplus()
        )

    def encode(
        self,
        x: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode input to latent distribution

        Args:
            x: [batch, 1, window_size]
            condition: [batch, 1, 2*condition_dim]

        Returns:
            mean: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        # Concatenate input and condition
        x_cond = torch.cat([x, condition], dim=-1)  # [batch, 1, window+2*cond]
        x_cond = x_cond.squeeze(1)  # [batch, window+2*cond]

        # Encoder
        hidden = self.encoder(x_cond)  # [batch, hidden_dim]

        # Latent parameters
        mean = self.fc_mean(hidden)  # [batch, latent_dim]
        logvar = self.fc_logvar(hidden)  # [batch, latent_dim]

        return mean, logvar

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick: z = μ + σ * ε, where ε ~ N(0,1)

        Args:
            mean: [batch, latent_dim]
            logvar: [batch, latent_dim]

        Returns:
            z: [batch, latent_dim]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + std * eps
        return z

    def decode(
        self,
        z: torch.Tensor,
        condition: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode latent to reconstruction distribution

        Args:
            z: [batch, latent_dim]
            condition: [batch, 1, 2*condition_dim]

        Returns:
            mu_x: [batch, 1, window_size] - Reconstruction mean
            logvar_x: [batch, 1, window_size] - Reconstruction variance
        """
        # Concatenate latent and condition
        condition = condition.squeeze(1)  # [batch, 2*cond_dim]
        z_cond = torch.cat([z, condition], dim=-1)  # [batch, latent+2*cond]

        # Decoder input
        hidden = self.decoder_input(z_cond)  # [batch, hidden_dim]
        hidden = hidden.unsqueeze(1)  # [batch, 1, hidden_dim]

        # Decoder
        recon = self.decoder(hidden)  # [batch, 1, window_size]

        # Reconstruction distribution parameters
        mu_x = self.fc_mu_x(recon)  # [batch, 1, window_size]
        logvar_x = self.fc_logvar_x(recon)  # [batch, 1, window_size]

        return mu_x, logvar_x

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: Encode → Reparameterize → Decode

        Args:
            x: [batch, 1, window_size]
            condition: [batch, 1, 2*condition_dim] (optional, will extract if None)

        Returns:
            mu_x: [batch, 1, window_size] - Reconstruction mean
            logvar_x: [batch, 1, window_size] - Reconstruction log-variance
            recon: [batch, 1, window_size] - Sampled reconstruction
            mean: [batch, latent_dim] - Latent mean
            logvar: [batch, latent_dim] - Latent log-variance
            z: [batch, latent_dim] - Sampled latent variable
        """
        # Extract condition if not provided
        if condition is None:
            condition = self.condition_extractor(x)

        # Encode
        mean, logvar = self.encode(x, condition)

        # Reparameterize
        z = self.reparameterize(mean, logvar)

        # Decode
        mu_x, logvar_x = self.decode(z, condition)

        # Sample reconstruction
        recon = self.reparameterize(mu_x, logvar_x)

        return mu_x, logvar_x, recon, mean, logvar, z

    def compute_loss(
        self,
        x: torch.Tensor,
        mu_x: torch.Tensor,
        logvar_x: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kl_weight: float = 0.005
    ) -> torch.Tensor:
        """
        Compute M-ELBO (Modified ELBO) loss
        - Reconstruction loss: Only on valid (non-anomalous, non-missing) data
        - KL divergence: Weighted by valid data ratio
        
        This implements the M-ELBO from the original FCVAE paper:
        M-ELBO = (1/|M|) Σ_{i∈M} [log p(x_i|z)] - (|M|/N) KL[q(z|x) || p(z)]

        Args:
            x: [batch, 1, window_size] - Original input
            mu_x: [batch, 1, window_size] - Reconstruction mean
            logvar_x: [batch, 1, window_size] - Reconstruction log-variance
            mean: [batch, latent_dim] - Latent mean
            logvar: [batch, latent_dim] - Latent log-variance
            z: [batch, latent_dim] - Sampled latent variable
            mask: [batch, window_size] - Mask for valid timesteps (1=use, 0=ignore)
                  Should exclude both anomalies and missing values
            kl_weight: KL divergence weight

        Returns:
            loss: Scalar loss value
        """
        # Squeeze dimensions for loss computation
        x = x.squeeze(1)  # [batch, window_size]
        mu_x = mu_x.squeeze(1)
        logvar_x = logvar_x.squeeze(1)

        # Apply mask if provided
        if mask is not None:
            mask = mask.float()  # [batch, window_size]
        else:
            mask = torch.ones_like(x)

        # === M-ELBO: Reconstruction Loss (Masked) ===
        # Only compute loss on valid (non-anomalous, non-missing) timesteps
        # -log N(x | μ_x, σ_x²) = 0.5 * (log(σ²) + (x-μ)²/σ²)
        var_x = torch.exp(logvar_x) + 1e-7  # Add small constant for stability
        recon_loss_per_point = 0.5 * (logvar_x + (x - mu_x) ** 2 / var_x)
        
        # Apply mask: only valid data contributes to loss
        masked_recon_loss = recon_loss_per_point * mask  # [batch, window_size]
        
        # Average over valid timesteps per sample, then over batch
        recon_loss = torch.mean(
            torch.mean(masked_recon_loss, dim=1),  # Mean over timesteps (already masked)
            dim=0
        )

        # === M-ELBO: KL Divergence (Weighted by valid ratio) ===
        # Weight KL by the ratio of valid data in each sample
        # m = |M| / N (valid ratio per sample)
        m = (mask.sum(dim=1, keepdim=True) / self.window_size).expand(-1, self.latent_dim)  # [batch, latent_dim]
        
        # KL divergence per dimension
        var = torch.exp(logvar) + 1e-7
        kl_per_dim = 0.5 * (z ** 2 - logvar - (z - mean) ** 2 / var)  # [batch, latent_dim]
        
        # Weight by valid ratio and average
        kl_loss = torch.mean(
            torch.mean(m * kl_per_dim, dim=1),  # Mean over latent dims (weighted)
            dim=0
        )

        # Total M-ELBO loss
        loss = recon_loss + kl_weight * kl_loss

        return loss

    def mcmc_inference(
        self,
        x: torch.Tensor,
        n_samples: int = 128
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        MCMC-based unsupervised inference for anomaly detection
        
        This method performs Monte Carlo sampling to estimate the reconstruction
        probability without using any label information. Used in test mode.
        
        Args:
            x: [batch, 1, window_size] - Input time series
            n_samples: Number of MCMC samples (default: 128)
        
        Returns:
            x: [batch, 1, window_size] - Input (unchanged)
            prob_all: [batch, 1, window_size] - Averaged log probability
        """
        origin_x = x.clone()
        
        # Extract frequency condition
        condition = self.condition_extractor(x)  # [batch, 1, 2*condition_dim]
        
        # Initialize probability accumulator
        prob_all = torch.zeros_like(x)
        
        # Monte Carlo sampling
        for _ in range(n_samples):
            # Encode
            mean, logvar = self.encode(x, condition)
            
            # Sample latent
            z = self.reparameterize(mean, logvar)
            
            # Decode
            mu_x, logvar_x = self.decode(z, condition)
            
            # Compute log probability: -0.5 * (log(σ²) + (x-μ)²/σ²)
            var_x = torch.exp(logvar_x) + 1e-7
            log_prob = -0.5 * (logvar_x + (origin_x - mu_x) ** 2 / var_x)
            
            # Accumulate
            prob_all += log_prob
        
        # Average over samples
        prob_all = prob_all / n_samples
        
        return origin_x, prob_all

    def compute_anomaly_score(
        self,
        x: torch.Tensor,
        n_samples: int = 128
    ) -> torch.Tensor:
        """
        Compute anomaly score using MCMC sampling

        Args:
            x: [batch, 1, window_size]
            n_samples: Number of MCMC samples

        Returns:
            scores: [batch, window_size] - Anomaly scores (higher = more anomalous)
        """
        self.eval()

        with torch.no_grad():
            # Extract condition
            condition = self.condition_extractor(x)

            # Encode
            mean, logvar = self.encode(x, condition)

            # MCMC sampling
            recon_probs = []
            for _ in range(n_samples):
                z = self.reparameterize(mean, logvar)
                mu_x, logvar_x = self.decode(z, condition)

                # Negative log-likelihood (higher = more anomalous)
                var_x = torch.exp(logvar_x) + 1e-7
                neg_log_prob = 0.5 * (logvar_x + (x - mu_x) ** 2 / var_x)
                recon_probs.append(neg_log_prob)

            # Average over samples
            scores = torch.stack(recon_probs).mean(dim=0)  # [batch, 1, window_size]
            scores = scores.squeeze(1)  # [batch, window_size]

        return scores


# ===== Test =====
if __name__ == '__main__':
    print("=" * 70)
    print("Testing Conditional VAE")
    print("=" * 70)

    # Hyperparameters
    batch_size = 32
    window_size = 96
    latent_dim = 8
    condition_dim = 16

    # Random input
    x = torch.randn(batch_size, 1, window_size)

    print(f"\nInput shape: {x.shape}")

    # Create model
    print("\n" + "-" * 70)
    print("Creating CVAE model...")
    print("-" * 70)
    model = ConditionalVAE(
        window_size=window_size,
        latent_dim=latent_dim,
        condition_dim=condition_dim,
        hidden_dim=100,
        d_model=256,
        d_ff=512,
        n_head=8,
        kernel_size=16,
        stride=8,
        dropout=0.05
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Forward pass
    print("\n" + "-" * 70)
    print("Testing forward pass...")
    print("-" * 70)
    mu_x, logvar_x, recon, mean, logvar = model(x)

    print(f"Reconstruction mean shape: {mu_x.shape}")
    print(f"Reconstruction logvar shape: {logvar_x.shape}")
    print(f"Sampled reconstruction shape: {recon.shape}")
    print(f"Latent mean shape: {mean.shape}")
    print(f"Latent logvar shape: {logvar.shape}")

    # Compute loss
    print("\n" + "-" * 70)
    print("Testing loss computation...")
    print("-" * 70)
    mask = torch.ones(batch_size, window_size)
    mask[:, -10:] = 0  # Mask last 10 timesteps
    loss = model.compute_loss(x, mu_x, logvar_x, mean, logvar, mask=mask)
    print(f"Loss: {loss.item():.4f}")

    # Compute anomaly scores
    print("\n" + "-" * 70)
    print("Testing anomaly score computation...")
    print("-" * 70)
    scores = model.compute_anomaly_score(x, n_samples=10)
    print(f"Anomaly scores shape: {scores.shape}")
    print(f"Score range: [{scores.min():.4f}, {scores.max():.4f}]")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)
