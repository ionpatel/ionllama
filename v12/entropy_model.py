"""
Entropy Model: Estimates next-byte prediction difficulty.

This small model predicts: "How hard is the next byte to predict?"
High entropy = start new patch (allocate more compute)
Low entropy = extend current patch (predictable, save compute)

Uses a lightweight architecture (Mamba-style or small Transformer)
to estimate entropy in O(n) time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


class MambaBlock(nn.Module):
    """
    Simplified Mamba-style block for efficient sequence modeling.
    
    Uses selective state space: input-dependent parameters for
    data-dependent sequence processing.
    
    This is a simplified version - real Mamba has more complex
    selective scan. We use Conv1D + gating as approximation.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Convolution for local context
        self.conv = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,  # Depthwise
        )
        
        # SSM parameters (simplified)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.d_inner, self.d_inner, bias=True)
        
        # A parameter (log-space for stability)
        self.A_log = nn.Parameter(torch.randn(self.d_inner, d_state))
        
        # D parameter (skip connection)
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        # Normalization
        self.norm = nn.RMSNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        residual = x
        
        # Input projection and split
        xz = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_in, z = xz.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Convolution (causal)
        x_conv = x_in.transpose(1, 2)  # (batch, d_inner, seq_len)
        x_conv = self.conv(x_conv)[:, :, :seq_len]  # Causal: truncate
        x_conv = x_conv.transpose(1, 2)  # (batch, seq_len, d_inner)
        x_conv = F.silu(x_conv)
        
        # Selective SSM (simplified)
        # In full Mamba, this is the selective scan
        # We approximate with position-wise gating + cumsum
        
        # Project to get B, C (state params)
        x_dbl = self.x_proj(x_conv)  # (batch, seq_len, d_state * 2)
        B, C = x_dbl.chunk(2, dim=-1)  # Each: (batch, seq_len, d_state)
        
        # Discretization step (simplified)
        dt = F.softplus(self.dt_proj(x_conv))  # (batch, seq_len, d_inner)
        
        # A matrix (discretized)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        
        # SSM computation (simplified - sequential for clarity)
        # In real Mamba: parallel selective scan
        # Here: exponentially decayed cumsum approximation
        
        # Decay: exp(A * dt)
        dA = torch.exp(A.unsqueeze(0).unsqueeze(0) * dt.unsqueeze(-1))
        # dA: (batch, seq_len, d_inner, d_state)
        
        # Input contribution
        dB_x = dt.unsqueeze(-1) * B.unsqueeze(2) * x_conv.unsqueeze(-1)
        # dB_x: (batch, seq_len, d_inner, d_state)
        
        # Simple approximation: just use the gated input
        # (Full Mamba uses selective scan here)
        y = (dA * dB_x).sum(dim=-1) + self.D * x_conv
        
        # Gate
        y = y * F.silu(z)
        
        # Output projection
        y = self.out_proj(y)
        
        # Residual
        y = self.norm(y + residual)
        
        return y


class EntropyEstimator(nn.Module):
    """
    Estimates next-byte prediction entropy.
    
    Architecture:
    - Byte embeddings (256)
    - 2-4 Mamba blocks (efficient O(n) processing)
    - Predict next byte → compute entropy from logits
    
    Output: entropy[i] = H(byte[i+1] | bytes[0:i])
    """
    
    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 2,
        d_state: int = 16,
        vocab_size: int = 256,  # Bytes
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Byte embeddings
        self.embed = nn.Embedding(vocab_size, d_model)
        
        # Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model=d_model, d_state=d_state)
            for _ in range(n_layers)
        ])
        
        # Output head (predict next byte)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Temperature for entropy computation
        self.register_buffer('log_vocab', torch.tensor(math.log(vocab_size)))
        
    def forward(
        self, 
        bytes_input: torch.Tensor,
        return_logits: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            bytes_input: (batch, seq_len) - byte values 0-255
            return_logits: If True, also return next-byte logits
        Returns:
            entropy: (batch, seq_len) - entropy at each position
            logits: (batch, seq_len, vocab) - if return_logits=True
        """
        # Embed bytes
        x = self.embed(bytes_input)  # (batch, seq_len, d_model)
        
        # Process through Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Predict next byte
        logits = self.head(x)  # (batch, seq_len, vocab_size)
        
        # Compute entropy from logits
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len)
        
        # Normalize to [0, 1] range (max entropy = log(vocab_size))
        entropy_normalized = entropy / self.log_vocab
        
        if return_logits:
            return entropy_normalized, logits
        return entropy_normalized
    
    def loss(
        self,
        bytes_input: torch.Tensor,
        target_bytes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Training loss: predict next byte.
        
        Args:
            bytes_input: (batch, seq_len) - input bytes
            target_bytes: (batch, seq_len) - target bytes (shifted by 1)
        Returns:
            loss: scalar
        """
        _, logits = self.forward(bytes_input, return_logits=True)
        
        # Flatten for cross-entropy
        logits_flat = logits.view(-1, self.vocab_size)
        target_flat = target_bytes.view(-1)
        
        loss = F.cross_entropy(logits_flat, target_flat)
        return loss


class DynamicPatcher(nn.Module):
    """
    Groups bytes into variable-length patches based on entropy.
    
    High entropy → start new patch (first byte of word, etc.)
    Low entropy → extend patch (predictable continuation)
    
    Output: List of patches, where each patch contains byte indices.
    """
    
    def __init__(
        self,
        entropy_model: EntropyEstimator,
        entropy_threshold: float = 0.5,  # Normalized [0, 1]
        min_patch_size: int = 1,
        max_patch_size: int = 8,
        pooling: str = "mean",  # How to pool bytes → patch
    ):
        super().__init__()
        
        self.entropy_model = entropy_model
        self.entropy_threshold = entropy_threshold
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.pooling = pooling
        
    def compute_patch_boundaries(
        self,
        bytes_input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute where patches start.
        
        Args:
            bytes_input: (batch, seq_len)
        Returns:
            is_boundary: (batch, seq_len) - True where new patch starts
            entropy: (batch, seq_len) - entropy values
        """
        batch, seq_len = bytes_input.shape
        
        # Get entropy estimates
        with torch.no_grad():
            entropy = self.entropy_model(bytes_input)
        
        # Initialize boundaries (first position is always a boundary)
        is_boundary = torch.zeros(batch, seq_len, dtype=torch.bool, device=bytes_input.device)
        is_boundary[:, 0] = True
        
        # Track patch sizes
        current_patch_size = torch.ones(batch, dtype=torch.long, device=bytes_input.device)
        
        for i in range(1, seq_len):
            # Start new patch if:
            # 1. Entropy exceeds threshold AND min patch size reached
            # 2. OR max patch size reached
            entropy_trigger = (entropy[:, i] > self.entropy_threshold) & \
                            (current_patch_size >= self.min_patch_size)
            size_trigger = current_patch_size >= self.max_patch_size
            
            new_boundary = entropy_trigger | size_trigger
            is_boundary[:, i] = new_boundary
            
            # Reset or increment patch size
            current_patch_size = torch.where(
                new_boundary,
                torch.ones_like(current_patch_size),
                current_patch_size + 1
            )
        
        return is_boundary, entropy
    
    def forward(
        self,
        bytes_input: torch.Tensor,
        byte_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert bytes to patches.
        
        Args:
            bytes_input: (batch, seq_len) - raw bytes
            byte_embeddings: (batch, seq_len, d_model) - embedded bytes
        Returns:
            patch_embeddings: (batch, n_patches, d_model)
            patch_lengths: (batch, n_patches) - length of each patch
            byte_to_patch: (batch, seq_len) - which patch each byte belongs to
        """
        batch, seq_len, d_model = byte_embeddings.shape
        
        # Get patch boundaries
        is_boundary, entropy = self.compute_patch_boundaries(bytes_input)
        
        # Convert boundaries to patch indices
        # Each byte gets assigned to a patch ID
        byte_to_patch = torch.cumsum(is_boundary.long(), dim=1) - 1
        # byte_to_patch: (batch, seq_len) values in [0, n_patches-1]
        
        # Count patches per batch
        n_patches = is_boundary.sum(dim=1)  # (batch,)
        max_patches = n_patches.max().item()
        
        # Pool bytes into patches
        # For each patch, gather its bytes and pool
        patch_embeddings = torch.zeros(
            batch, max_patches, d_model, 
            device=byte_embeddings.device,
            dtype=byte_embeddings.dtype
        )
        patch_lengths = torch.zeros(
            batch, max_patches,
            device=byte_embeddings.device,
            dtype=torch.long
        )
        
        for b in range(batch):
            for p in range(n_patches[b]):
                # Find bytes belonging to this patch
                mask = byte_to_patch[b] == p
                patch_bytes = byte_embeddings[b, mask]  # (patch_len, d_model)
                
                # Pool
                if self.pooling == "mean":
                    patch_embeddings[b, p] = patch_bytes.mean(dim=0)
                elif self.pooling == "last":
                    patch_embeddings[b, p] = patch_bytes[-1]
                elif self.pooling == "first":
                    patch_embeddings[b, p] = patch_bytes[0]
                else:
                    patch_embeddings[b, p] = patch_bytes.mean(dim=0)
                
                patch_lengths[b, p] = mask.sum()
        
        return patch_embeddings, patch_lengths, byte_to_patch
    
    def unpatch(
        self,
        patch_embeddings: torch.Tensor,
        byte_to_patch: torch.Tensor,
        seq_len: int,
    ) -> torch.Tensor:
        """
        Expand patches back to byte-level.
        
        Args:
            patch_embeddings: (batch, n_patches, d_model)
            byte_to_patch: (batch, seq_len) - mapping from bytes to patches
            seq_len: Original sequence length
        Returns:
            byte_embeddings: (batch, seq_len, d_model)
        """
        batch, n_patches, d_model = patch_embeddings.shape
        
        # Gather patch embedding for each byte position
        byte_embeddings = torch.gather(
            patch_embeddings,
            dim=1,
            index=byte_to_patch.unsqueeze(-1).expand(-1, -1, d_model)
        )
        
        return byte_embeddings


# Test
if __name__ == "__main__":
    print("Testing Entropy Model and Patcher...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test entropy model
    print("\n--- Entropy Model ---")
    entropy_model = EntropyEstimator(d_model=128, n_layers=2).to(device)
    
    # Create sample bytes
    text = "Hello, how are you today?"
    bytes_input = torch.tensor([[ord(c) for c in text]], device=device)
    print(f"Input: '{text}'")
    print(f"Bytes: {bytes_input.shape}")
    
    entropy = entropy_model(bytes_input)
    print(f"Entropy shape: {entropy.shape}")
    print(f"Entropy values: {entropy[0].tolist()[:10]}...")  # First 10
    
    # Parameter count
    params = sum(p.numel() for p in entropy_model.parameters())
    print(f"Entropy model params: {params:,}")
    
    # Test dynamic patcher
    print("\n--- Dynamic Patcher ---")
    patcher = DynamicPatcher(
        entropy_model=entropy_model,
        entropy_threshold=0.4,
        max_patch_size=6,
    ).to(device)
    
    # Get boundaries
    is_boundary, _ = patcher.compute_patch_boundaries(bytes_input)
    boundaries = is_boundary[0].nonzero().squeeze().tolist()
    print(f"Patch boundaries at: {boundaries}")
    
    # Visualize patches
    patches = []
    current = ""
    for i, c in enumerate(text):
        if i in boundaries and current:
            patches.append(current)
            current = c
        else:
            current += c
    patches.append(current)
    print(f"Patches: {patches}")
    print(f"Number of patches: {len(patches)}")
    
    # Test full patching
    byte_embeddings = entropy_model.embed(bytes_input)  # Reuse embeddings
    patch_emb, patch_lens, byte_to_patch = patcher(bytes_input, byte_embeddings)
    print(f"Patch embeddings: {patch_emb.shape}")
    print(f"Patch lengths: {patch_lens[0].tolist()[:10]}")
    
    # Test unpatching
    reconstructed = patcher.unpatch(patch_emb, byte_to_patch, len(text))
    print(f"Reconstructed: {reconstructed.shape}")
    
    print("\n✅ Entropy Model and Patcher tests passed!")
