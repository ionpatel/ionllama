"""
IntegralLNN: Context Accumulation via Integration

The key insight: Language ACCUMULATES meaning, it doesn't measure instantaneous change.
- Standard LNN: dh/dt = -h/τ + f(h, x)  (differential - measures change)
- IntegralLNN: h = ∫f(h, x)dt           (integral - accumulates context)

Implementation uses Conv1D + cumsum for O(n) efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class IntegralLNNLayer(nn.Module):
    """
    Single IntegralLNN layer with gated integration.
    
    For each position i, computes:
        h[i] = Σ(j=0 to i) decay[i-j] * gate[j] * candidate[j]
    
    This is integration with exponential decay (prevents explosion).
    """
    
    def __init__(
        self,
        d_model: int,
        kernel_size: int = 7,
        decay_init: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.kernel_size = kernel_size
        
        # Gating mechanism (what to integrate)
        self.gate_proj = nn.Conv1d(
            d_model, d_model, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2,
            groups=d_model  # Depthwise for efficiency
        )
        
        # Candidate values (what values to accumulate)
        self.candidate_proj = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model
        )
        
        # Learnable decay rate per dimension
        self.log_decay = nn.Parameter(
            torch.full((d_model,), math.log(decay_init))
        )
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Normalization and dropout
        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = x.shape
        residual = x
        
        # Transpose for Conv1d: (batch, d_model, seq_len)
        x_t = x.transpose(1, 2)
        
        # Compute gate and candidate
        gate = torch.sigmoid(self.gate_proj(x_t))      # (batch, d_model, seq_len)
        candidate = torch.tanh(self.candidate_proj(x_t))  # (batch, d_model, seq_len)
        
        # Gated values to integrate
        gated = gate * candidate  # (batch, d_model, seq_len)
        
        # Compute decay weights: exp(-α * distance)
        # decay[i] = exp(-α * i) for position offset i
        decay_rate = torch.exp(self.log_decay)  # (d_model,)
        positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)
        
        # For efficiency, we use a causal convolution approach
        # Instead of true cumsum with decay (which is O(n²) naive),
        # we use the recurrence: h[i] = decay * h[i-1] + gated[i]
        
        # This is equivalent to exponentially-weighted cumsum
        h = self._parallel_scan(gated, decay_rate)
        
        # Transpose back: (batch, seq_len, d_model)
        h = h.transpose(1, 2)
        
        # Output projection with residual
        out = self.out_proj(h)
        out = self.dropout(out)
        out = self.norm(out + residual)
        
        return out
    
    def _parallel_scan(
        self, 
        x: torch.Tensor, 
        decay: torch.Tensor
    ) -> torch.Tensor:
        """
        Parallel scan for exponentially-decayed cumsum.
        
        Computes: h[i] = Σ(j=0 to i) decay^(i-j) * x[j]
        
        Using the recurrence: h[i] = decay * h[i-1] + x[i]
        
        This is O(n) via sequential scan or O(n log n) via parallel scan.
        For simplicity, we use sequential (still very fast on GPU).
        
        Args:
            x: (batch, d_model, seq_len)
            decay: (d_model,) - decay rate per dimension
        Returns:
            (batch, d_model, seq_len)
        """
        batch, d_model, seq_len = x.shape
        
        # Reshape decay for broadcasting
        decay = decay.view(1, d_model, 1)
        
        # Use torch's built-in for efficiency when available
        # Otherwise, manual sequential scan
        
        # Method: Exponentially-weighted moving average via conv
        # This approximates the parallel scan efficiently
        
        # For short sequences, direct cumsum with decay
        if seq_len <= 512:
            # Build decay matrix: decay[i,j] = decay^(i-j) if i >= j else 0
            # This is O(n²) memory but fast for short sequences
            positions = torch.arange(seq_len, device=x.device)
            distances = positions.unsqueeze(0) - positions.unsqueeze(1)  # (seq, seq)
            distances = distances.clamp(min=0)
            
            # Causal mask
            mask = (distances >= 0).float()
            mask = torch.tril(mask)
            
            # Decay weights: (d_model, seq, seq)
            decay_weights = decay.squeeze(0).unsqueeze(1) ** distances.unsqueeze(0).float()
            decay_weights = decay_weights * mask.unsqueeze(0)
            
            # Apply: h = decay_weights @ x
            # (batch, d_model, seq, seq) @ (batch, d_model, seq, 1)
            h = torch.einsum('dij,bdj->bdi', decay_weights, x)
            
            return h
        
        else:
            # For long sequences, use sequential scan (still O(n))
            h = torch.zeros_like(x)
            h[:, :, 0] = x[:, :, 0]
            
            for i in range(1, seq_len):
                h[:, :, i] = decay.squeeze(-1) * h[:, :, i-1] + x[:, :, i]
            
            return h


class IntegralLNNBlock(nn.Module):
    """
    Full IntegralLNN block with FFN.
    
    IntegralLNN → FFN (SwiGLU) → Output
    """
    
    def __init__(
        self,
        d_model: int,
        ffn_mult: float = 2.67,  # SwiGLU optimal
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.lnn = IntegralLNNLayer(
            d_model=d_model,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        
        # SwiGLU FFN
        ffn_dim = int(d_model * ffn_mult)
        self.ffn_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.ffn_up = nn.Linear(d_model, ffn_dim, bias=False)
        self.ffn_down = nn.Linear(ffn_dim, d_model, bias=False)
        
        self.ffn_norm = nn.RMSNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        # IntegralLNN
        x = self.lnn(x)
        
        # SwiGLU FFN
        residual = x
        gate = F.silu(self.ffn_gate(x))
        up = self.ffn_up(x)
        x = gate * up
        x = self.ffn_down(x)
        x = self.ffn_dropout(x)
        x = self.ffn_norm(x + residual)
        
        return x


class IntegralLNNEncoder(nn.Module):
    """
    Stack of IntegralLNN blocks for local encoding.
    
    Used in ionLlama v12 for:
    - Local Encoder: bytes → patch representations
    - Local Decoder: patch outputs → byte representations
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        kernel_size: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            IntegralLNNBlock(
                d_model=d_model,
                kernel_size=kernel_size,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        return x


# Test
if __name__ == "__main__":
    print("Testing IntegralLNN...")
    
    batch, seq_len, d_model = 2, 128, 256
    x = torch.randn(batch, seq_len, d_model)
    
    # Test single layer
    layer = IntegralLNNLayer(d_model=d_model)
    out = layer(x)
    print(f"Single layer: {x.shape} → {out.shape}")
    
    # Test block
    block = IntegralLNNBlock(d_model=d_model)
    out = block(x)
    print(f"Block: {x.shape} → {out.shape}")
    
    # Test encoder
    encoder = IntegralLNNEncoder(d_model=d_model, n_layers=2)
    out = encoder(x)
    print(f"Encoder (2 layers): {x.shape} → {out.shape}")
    
    # Parameter count
    params = sum(p.numel() for p in encoder.parameters())
    print(f"Encoder params: {params:,}")
    
    print("✅ IntegralLNN tests passed!")
