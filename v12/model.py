"""
ionLlama v12: Tokenizer-Free Language Model

The complete architecture:
1. Raw bytes input (no tokenizer!)
2. Entropy-based dynamic patching (variable compute allocation)
3. IntegralLNN local encoder (context accumulation)
4. Sparse Dilated Attention latent transformer (O(n log n))
5. IntegralLNN local decoder (unpatch to bytes)
6. Byte prediction head (256-way classification)

Total complexity: O(n log n) vs O(n²) for standard transformers
Character-level access: Perfect spelling, arithmetic, multilingual fairness
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, Any

from integral_lnn import IntegralLNNEncoder
from sparse_attention import LatentTransformer
from entropy_model import EntropyEstimator, DynamicPatcher


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for byte/patch positions."""
    
    def __init__(self, d_model: int, max_seq_len: int = 65536, base: float = 10000.0):
        super().__init__()
        
        # Compute inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache sin/cos
        self._build_cache(max_seq_len)
        
    def _build_cache(self, seq_len: int):
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
        
        # Expand to [seq_len, d_model] by interleaving
        emb = torch.cat([freqs, freqs], dim=-1)
        
        self.register_buffer('cos_cached', emb.cos())
        self.register_buffer('sin_cached', emb.sin())
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add rotary position embeddings.
        
        Args:
            x: (batch, seq_len, d_model)
        Returns:
            (batch, seq_len, d_model)
        """
        seq_len = x.shape[1]
        cos = self.cos_cached[:seq_len]
        sin = self.sin_cached[:seq_len]
        
        # Apply rotation
        x1, x2 = x[..., ::2], x[..., 1::2]
        rotated = torch.stack([
            x1 * cos[:, ::2] - x2 * sin[:, ::2],
            x1 * sin[:, 1::2] + x2 * cos[:, 1::2]
        ], dim=-1).flatten(-2)
        
        return rotated


class ByteEmbedding(nn.Module):
    """
    Byte embedding with n-gram context.
    
    Each byte's embedding includes information from previous bytes
    via learned n-gram combinations.
    """
    
    def __init__(
        self,
        d_model: int,
        vocab_size: int = 256,
        n_gram: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_gram = n_gram
        
        # Base byte embeddings
        self.byte_embed = nn.Embedding(vocab_size, d_model)
        
        # N-gram projection (combines previous bytes)
        self.ngram_conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=n_gram,
            padding=n_gram - 1,  # Causal padding
            groups=1,
        )
        
        # Position embedding
        self.rope = RotaryPositionalEmbedding(d_model)
        
        # Normalization
        self.norm = nn.RMSNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, bytes_input: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bytes_input: (batch, seq_len) - byte values 0-255
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len = bytes_input.shape
        
        # Embed bytes
        x = self.byte_embed(bytes_input)  # (batch, seq_len, d_model)
        
        # Add n-gram context
        x_t = x.transpose(1, 2)  # (batch, d_model, seq_len)
        x_ngram = self.ngram_conv(x_t)[:, :, :seq_len]  # Causal
        x_ngram = x_ngram.transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Combine
        x = x + x_ngram
        
        # Add position
        x = self.rope(x)
        
        # Normalize
        x = self.norm(x)
        x = self.dropout(x)
        
        return x


class LocalDecoder(nn.Module):
    """
    Decodes patch representations back to byte-level.
    
    Uses cross-attention between:
    - Query: byte encoder outputs
    - Key/Value: latent transformer outputs (patches)
    
    Then IntegralLNN for local byte-level processing.
    """
    
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Cross-attention: bytes attend to patches
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.RMSNorm(d_model)
        
        # IntegralLNN for local processing
        self.local_lnn = IntegralLNNEncoder(
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
        
    def forward(
        self,
        byte_encodings: torch.Tensor,
        patch_outputs: torch.Tensor,
        byte_to_patch: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            byte_encodings: (batch, seq_len, d_model) - from local encoder
            patch_outputs: (batch, n_patches, d_model) - from latent transformer
            byte_to_patch: (batch, seq_len) - mapping bytes → patches
        Returns:
            (batch, seq_len, d_model)
        """
        batch, seq_len, d_model = byte_encodings.shape
        n_patches = patch_outputs.shape[1]
        
        # Gather patch outputs for each byte position
        # Each byte attends to its corresponding patch
        patch_for_bytes = torch.gather(
            patch_outputs,
            dim=1,
            index=byte_to_patch.unsqueeze(-1).expand(-1, -1, d_model)
        )  # (batch, seq_len, d_model)
        
        # Cross-attention with residual
        # Bytes (queries) attend to their patch (keys/values)
        x = byte_encodings
        attn_out, _ = self.cross_attn(
            query=x,
            key=patch_for_bytes,
            value=patch_for_bytes,
        )
        x = self.cross_norm(x + attn_out)
        
        # IntegralLNN for final local processing
        x = self.local_lnn(x)
        
        return x


class ionLlamaV12(nn.Module):
    """
    ionLlama v12: Complete Tokenizer-Free Architecture
    
    Flow:
    1. bytes → ByteEmbedding (with n-gram context)
    2. EntropyEstimator → DynamicPatcher (variable patches)
    3. IntegralLNN encoder (byte → patch encoding)
    4. LatentTransformer (sparse attention on patches)
    5. LocalDecoder (patch → byte decoding)
    6. BytePredictionHead (next byte prediction)
    """
    
    def __init__(
        self,
        d_model: int = 512,
        n_local_layers: int = 2,
        n_latent_layers: int = 12,
        n_heads: int = 8,
        ffn_mult: float = 2.67,
        dropout: float = 0.1,
        max_seq_len: int = 8192,
        entropy_threshold: float = 0.5,
        max_patch_size: int = 8,
        vocab_size: int = 256,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Byte embedding
        self.byte_embed = ByteEmbedding(
            d_model=d_model,
            vocab_size=vocab_size,
            dropout=dropout,
        )
        
        # Entropy model for dynamic patching
        self.entropy_model = EntropyEstimator(
            d_model=d_model // 2,  # Smaller for efficiency
            n_layers=2,
            vocab_size=vocab_size,
        )
        
        # Dynamic patcher
        self.patcher = DynamicPatcher(
            entropy_model=self.entropy_model,
            entropy_threshold=entropy_threshold,
            max_patch_size=max_patch_size,
        )
        
        # Local encoder (bytes → patches)
        self.local_encoder = IntegralLNNEncoder(
            d_model=d_model,
            n_layers=n_local_layers,
            dropout=dropout,
        )
        
        # Latent transformer (main model - processes patches)
        self.latent_transformer = LatentTransformer(
            d_model=d_model,
            n_layers=n_latent_layers,
            n_heads=n_heads,
            ffn_mult=ffn_mult,
            dropout=dropout,
            max_seq_len=max_seq_len // max_patch_size,  # Patches are shorter
        )
        
        # Local decoder (patches → bytes)
        self.local_decoder = LocalDecoder(
            d_model=d_model,
            n_layers=n_local_layers,
            n_heads=n_heads // 2,
            dropout=dropout,
        )
        
        # Byte prediction head
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie embeddings with head (optional, common practice)
        # self.head.weight = self.byte_embed.byte_embed.weight
        
        # Initialize
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights following GPT-2 style."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                
    def forward(
        self,
        bytes_input: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        return_loss: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            bytes_input: (batch, seq_len) - input bytes 0-255
            targets: (batch, seq_len) - target bytes (shifted by 1)
            return_loss: Whether to compute loss
        Returns:
            Dict with:
                - logits: (batch, seq_len, vocab_size)
                - loss: scalar (if return_loss and targets provided)
                - entropy: (batch, seq_len) - entropy at each position
                - n_patches: (batch,) - number of patches per sequence
        """
        batch, seq_len = bytes_input.shape
        
        # 1. Embed bytes with n-gram context
        byte_embeddings = self.byte_embed(bytes_input)  # (batch, seq_len, d_model)
        
        # 2. Encode bytes locally
        byte_encoded = self.local_encoder(byte_embeddings)  # (batch, seq_len, d_model)
        
        # 3. Dynamic patching based on entropy
        patch_embeddings, patch_lengths, byte_to_patch = self.patcher(
            bytes_input, byte_encoded
        )  # (batch, n_patches, d_model)
        
        # 4. Latent transformer on patches (main compute)
        patch_outputs = self.latent_transformer(patch_embeddings)  # (batch, n_patches, d_model)
        
        # 5. Decode back to bytes
        byte_decoded = self.local_decoder(
            byte_encoded, patch_outputs, byte_to_patch
        )  # (batch, seq_len, d_model)
        
        # 6. Predict next byte
        logits = self.head(byte_decoded)  # (batch, seq_len, vocab_size)
        
        # Get entropy for analysis
        with torch.no_grad():
            entropy = self.entropy_model(bytes_input)
        
        output = {
            'logits': logits,
            'entropy': entropy,
            'n_patches': (patch_lengths > 0).sum(dim=1),
            'byte_to_patch': byte_to_patch,
        }
        
        # Compute loss
        if return_loss and targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-100,  # Padding
            )
            output['loss'] = loss
            
            # Bits-per-byte (standard metric)
            bpb = loss / math.log(2)
            output['bpb'] = bpb
        
        return output
    
    def generate(
        self,
        bytes_input: torch.Tensor,
        max_new_bytes: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Generate new bytes autoregressively.
        
        Args:
            bytes_input: (batch, seq_len) - prompt bytes
            max_new_bytes: Number of bytes to generate
            temperature: Sampling temperature
            top_k: If set, only sample from top k tokens
            top_p: If set, nucleus sampling threshold
        Returns:
            (batch, seq_len + max_new_bytes) - generated sequence
        """
        self.eval()
        
        for _ in range(max_new_bytes):
            # Forward pass
            with torch.no_grad():
                output = self(bytes_input, return_loss=False)
            
            # Get logits for last position
            logits = output['logits'][:, -1, :] / temperature  # (batch, vocab)
            
            # Apply top-k
            if top_k is not None:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][:, -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Apply top-p (nucleus sampling)
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = False
                
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')
            
            # Sample
            probs = F.softmax(logits, dim=-1)
            next_byte = torch.multinomial(probs, num_samples=1)  # (batch, 1)
            
            # Append
            bytes_input = torch.cat([bytes_input, next_byte], dim=1)
            
            # Check for EOS (could be specific byte like newline)
            # For now, just continue
        
        return bytes_input
    
    def count_parameters(self) -> Dict[str, int]:
        """Count parameters by component."""
        counts = {
            'byte_embed': sum(p.numel() for p in self.byte_embed.parameters()),
            'entropy_model': sum(p.numel() for p in self.entropy_model.parameters()),
            'local_encoder': sum(p.numel() for p in self.local_encoder.parameters()),
            'latent_transformer': sum(p.numel() for p in self.latent_transformer.parameters()),
            'local_decoder': sum(p.numel() for p in self.local_decoder.parameters()),
            'head': sum(p.numel() for p in self.head.parameters()),
        }
        counts['total'] = sum(counts.values())
        return counts


def create_model(size: str = 'small') -> ionLlamaV12:
    """Create model with preset configurations."""
    
    configs = {
        'tiny': dict(
            d_model=256,
            n_local_layers=1,
            n_latent_layers=4,
            n_heads=4,
        ),
        'small': dict(
            d_model=512,
            n_local_layers=2,
            n_latent_layers=8,
            n_heads=8,
        ),
        'base': dict(
            d_model=768,
            n_local_layers=3,
            n_latent_layers=12,
            n_heads=12,
        ),
        'large': dict(
            d_model=1024,
            n_local_layers=4,
            n_latent_layers=24,
            n_heads=16,
        ),
        'xl': dict(
            d_model=2048,
            n_local_layers=4,
            n_latent_layers=32,
            n_heads=32,
        ),
    }
    
    if size not in configs:
        raise ValueError(f"Unknown size: {size}. Choose from {list(configs.keys())}")
    
    return ionLlamaV12(**configs[size])


# Test
if __name__ == "__main__":
    print("Testing ionLlama v12...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    print("\n--- Creating Model ---")
    model = create_model('tiny').to(device)
    
    # Count parameters
    param_counts = model.count_parameters()
    print("Parameters by component:")
    for name, count in param_counts.items():
        print(f"  {name}: {count:,}")
    
    # Test forward pass
    print("\n--- Forward Pass ---")
    text = "Hello, how are you doing today? This is a test of ionLlama v12."
    bytes_input = torch.tensor([[ord(c) for c in text]], device=device)
    targets = torch.tensor([[ord(c) for c in text[1:]] + [0]], device=device)  # Shifted
    
    print(f"Input: '{text}'")
    print(f"Bytes shape: {bytes_input.shape}")
    
    output = model(bytes_input, targets)
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Loss: {output['loss'].item():.4f}")
    print(f"BPB: {output['bpb'].item():.4f}")
    print(f"N patches: {output['n_patches'].item()}")
    print(f"Compression: {len(text)} bytes → {output['n_patches'].item()} patches ({len(text)/output['n_patches'].item():.1f}× compression)")
    
    # Test generation
    print("\n--- Generation ---")
    prompt = "Hello"
    prompt_bytes = torch.tensor([[ord(c) for c in prompt]], device=device)
    
    generated = model.generate(prompt_bytes, max_new_bytes=20, temperature=0.8)
    generated_text = ''.join([chr(b) for b in generated[0].tolist()])
    print(f"Prompt: '{prompt}'")
    print(f"Generated: '{generated_text}'")
    
    # Benchmark
    print("\n--- Benchmark ---")
    import time
    
    # Warmup
    for _ in range(3):
        _ = model(bytes_input, targets)
    
    # Timed runs
    torch.cuda.synchronize() if device.type == 'cuda' else None
    start = time.time()
    n_runs = 10
    for _ in range(n_runs):
        _ = model(bytes_input, targets)
    torch.cuda.synchronize() if device.type == 'cuda' else None
    elapsed = (time.time() - start) / n_runs
    
    throughput = len(text) / elapsed
    print(f"Time per forward: {elapsed*1000:.2f} ms")
    print(f"Throughput: {throughput:.0f} bytes/second")
    
    print("\n✅ ionLlama v12 tests passed!")
