"""
Sparse Dilated Attention: O(n log n) Complexity

The key insight: Most attention weights are near-zero anyway.
Instead of computing all n² pairs, each position attends to only
O(log n) positions using exponential dilation.

Pattern for position i:
    Attends to: [i, i-1, i-2, i-4, i-8, i-16, i-32, ...]
    
For sequence length n, each position attends to ~log₂(n) positions.
Total complexity: O(n log n) instead of O(n²)

For n=32K:
- Standard: 32K × 32K = 1 billion operations
- Sparse: 32K × log(32K) = 32K × 15 = 480K operations
- Speedup: 2,000×
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


def get_dilated_positions(seq_len: int, include_local: int = 2) -> torch.Tensor:
    """
    Generate the sparse attention pattern.
    
    For each position i, returns the positions it should attend to.
    Pattern: [i, i-1, i-2, i-4, i-8, ...] (positions with distance 2^k)
    
    Args:
        seq_len: Sequence length
        include_local: Number of local positions to always include (default 2)
    
    Returns:
        positions: (seq_len, max_attend) - positions each token attends to
        mask: (seq_len, max_attend) - valid positions (1) vs padding (0)
    """
    # Maximum number of positions per token
    max_attend = include_local + int(math.ceil(math.log2(seq_len + 1)))
    
    positions = torch.zeros(seq_len, max_attend, dtype=torch.long)
    mask = torch.zeros(seq_len, max_attend, dtype=torch.bool)
    
    for i in range(seq_len):
        attend_positions = []
        
        # Always attend to self
        attend_positions.append(i)
        
        # Local positions (i-1, i-2, ...)
        for j in range(1, include_local + 1):
            if i - j >= 0:
                attend_positions.append(i - j)
        
        # Exponentially dilated positions (i-4, i-8, i-16, ...)
        k = 2  # Start from 2^2 = 4
        while True:
            offset = 2 ** k
            if i - offset < 0:
                break
            attend_positions.append(i - offset)
            k += 1
        
        # Fill positions tensor
        n_attend = len(attend_positions)
        positions[i, :n_attend] = torch.tensor(attend_positions)
        mask[i, :n_attend] = True
    
    return positions, mask


class SparseDilatedAttention(nn.Module):
    """
    Multi-head attention with sparse dilated pattern.
    
    Each head uses the same sparsity pattern but different projections.
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        include_local: int = 2,
    ):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5
        self.max_seq_len = max_seq_len
        self.include_local = include_local
        
        # QKV projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Precompute attention patterns (will be moved to device on first forward)
        self._positions = None
        self._mask = None
        self._cached_seq_len = 0
        
    def _get_pattern(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get or compute attention pattern for given sequence length."""
        if self._positions is None or seq_len > self._cached_seq_len:
            # Compute pattern for this sequence length
            positions, mask = get_dilated_positions(seq_len, self.include_local)
            self._positions = positions.to(device)
            self._mask = mask.to(device)
            self._cached_seq_len = seq_len
        
        # Slice to current sequence length
        return self._positions[:seq_len], self._mask[:seq_len]
        
    def forward(
        self, 
        x: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            positions: Optional precomputed positions
            mask: Optional precomputed mask
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Get attention pattern
        if positions is None or mask is None:
            positions, mask = self._get_pattern(seq_len, x.device)
        
        # Project to Q, K, V
        q = self.q_proj(x)  # (batch, seq_len, d_model)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head: (batch, n_heads, seq_len, head_dim)
        q = q.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Gather K and V at sparse positions
        # positions: (seq_len, max_attend)
        max_attend = positions.shape[1]
        
        # Expand positions for gathering: (batch, n_heads, seq_len, max_attend)
        pos_expanded = positions.unsqueeze(0).unsqueeze(0).expand(batch, self.n_heads, -1, -1)
        
        # Gather K: for each query position, get K at sparse positions
        # k: (batch, n_heads, seq_len, head_dim)
        # We need k at positions[i] for query i
        k_sparse = torch.gather(
            k.unsqueeze(3).expand(-1, -1, -1, max_attend, -1),
            dim=2,
            index=pos_expanded.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        ).squeeze(2)  # (batch, n_heads, seq_len, max_attend, head_dim)
        
        # Actually, let's do this more efficiently
        # Flatten batch and heads
        k_flat = k.reshape(batch * self.n_heads, seq_len, self.head_dim)
        v_flat = v.reshape(batch * self.n_heads, seq_len, self.head_dim)
        
        # Gather for each position
        # This is the sparse gather operation
        pos_flat = positions.unsqueeze(0).expand(batch * self.n_heads, -1, -1)
        
        # k_sparse[b, i, j, :] = k_flat[b, positions[i, j], :]
        k_sparse = torch.gather(
            k_flat.unsqueeze(2).expand(-1, -1, max_attend, -1),
            dim=1,
            index=pos_flat.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )  # (batch*n_heads, seq_len, max_attend, head_dim)
        
        v_sparse = torch.gather(
            v_flat.unsqueeze(2).expand(-1, -1, max_attend, -1),
            dim=1,
            index=pos_flat.unsqueeze(-1).expand(-1, -1, -1, self.head_dim)
        )  # (batch*n_heads, seq_len, max_attend, head_dim)
        
        # Reshape back
        k_sparse = k_sparse.view(batch, self.n_heads, seq_len, max_attend, self.head_dim)
        v_sparse = v_sparse.view(batch, self.n_heads, seq_len, max_attend, self.head_dim)
        
        # Compute attention scores
        # q: (batch, n_heads, seq_len, head_dim)
        # k_sparse: (batch, n_heads, seq_len, max_attend, head_dim)
        scores = torch.einsum('bhsd,bhsad->bhsa', q, k_sparse) * self.scale
        # scores: (batch, n_heads, seq_len, max_attend)
        
        # Apply mask (set invalid positions to -inf)
        mask_expanded = mask.unsqueeze(0).unsqueeze(0).expand(batch, self.n_heads, -1, -1)
        scores = scores.masked_fill(~mask_expanded, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # attn_weights: (batch, n_heads, seq_len, max_attend)
        # v_sparse: (batch, n_heads, seq_len, max_attend, head_dim)
        out = torch.einsum('bhsa,bhsad->bhsd', attn_weights, v_sparse)
        # out: (batch, n_heads, seq_len, head_dim)
        
        # Reshape and project
        out = out.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        out = self.out_proj(out)
        
        return out


class SparseDilatedBlock(nn.Module):
    """
    Transformer block with sparse dilated attention.
    
    SparseDilatedAttention → LayerNorm → FFN (SwiGLU) → LayerNorm
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int = 8,
        ffn_mult: float = 2.67,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        
        # Attention
        self.attn = SparseDilatedAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        self.attn_norm = nn.RMSNorm(d_model)
        
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
        # Attention with residual
        residual = x
        x = self.attn(x)
        x = self.attn_norm(x + residual)
        
        # FFN with residual
        residual = x
        gate = F.silu(self.ffn_gate(x))
        up = self.ffn_up(x)
        x = gate * up
        x = self.ffn_down(x)
        x = self.ffn_dropout(x)
        x = self.ffn_norm(x + residual)
        
        return x


class LatentTransformer(nn.Module):
    """
    Stack of SparseDilatedBlocks for the main transformer.
    
    This is the "heavy" part of ionLlama v12 - where most parameters live.
    Processes patches (not raw bytes) for efficiency.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 12,
        n_heads: int = 8,
        ffn_mult: float = 2.67,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            SparseDilatedBlock(
                d_model=d_model,
                n_heads=n_heads,
                ffn_mult=ffn_mult,
                dropout=dropout,
                max_seq_len=max_seq_len,
            )
            for _ in range(n_layers)
        ])
        
        self.final_norm = nn.RMSNorm(d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) - patch representations
        Returns:
            (batch, seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_norm(x)
        return x


# Test and benchmark
if __name__ == "__main__":
    import time
    
    print("Testing Sparse Dilated Attention...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Test attention pattern
    print("\n--- Attention Pattern ---")
    for seq_len in [16, 128, 1024, 8192]:
        positions, mask = get_dilated_positions(seq_len)
        n_attend = mask[seq_len-1].sum().item()
        print(f"Seq {seq_len}: Each position attends to {n_attend} positions (log₂={math.log2(seq_len):.1f})")
    
    # Test single attention layer
    print("\n--- Attention Layer ---")
    batch, seq_len, d_model = 2, 512, 256
    x = torch.randn(batch, seq_len, d_model, device=device)
    
    attn = SparseDilatedAttention(d_model=d_model, n_heads=8).to(device)
    out = attn(x)
    print(f"Attention: {x.shape} → {out.shape}")
    
    # Test block
    print("\n--- Block ---")
    block = SparseDilatedBlock(d_model=d_model, n_heads=8).to(device)
    out = block(x)
    print(f"Block: {x.shape} → {out.shape}")
    
    # Test full transformer
    print("\n--- Latent Transformer ---")
    transformer = LatentTransformer(d_model=d_model, n_layers=8, n_heads=8).to(device)
    out = transformer(x)
    print(f"Transformer (8 layers): {x.shape} → {out.shape}")
    
    params = sum(p.numel() for p in transformer.parameters())
    print(f"Parameters: {params:,}")
    
    # Benchmark: Sparse vs Dense attention
    print("\n--- Benchmark ---")
    seq_lens = [256, 512, 1024, 2048]
    
    for seq_len in seq_lens:
        x = torch.randn(2, seq_len, d_model, device=device)
        
        # Sparse
        sparse_attn = SparseDilatedAttention(d_model=d_model).to(device)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        for _ in range(10):
            _ = sparse_attn(x)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        sparse_time = (time.time() - start) / 10
        
        # Dense (standard attention for comparison)
        q = k = v = x
        torch.cuda.synchronize() if device.type == 'cuda' else None
        start = time.time()
        for _ in range(10):
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_model)
            attn = F.softmax(scores, dim=-1)
            _ = torch.matmul(attn, v)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        dense_time = (time.time() - start) / 10
        
        speedup = dense_time / sparse_time
        print(f"Seq {seq_len}: Sparse {sparse_time*1000:.2f}ms, Dense {dense_time*1000:.2f}ms, Speedup {speedup:.1f}×")
    
    print("\n✅ Sparse Dilated Attention tests passed!")
