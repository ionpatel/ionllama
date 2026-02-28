# ionLlama v12: Tokenizer-Free Architecture

## Vision
A byte-level language model that combines:
- **No tokenizer** — Direct byte processing (0-255)
- **Dynamic patching** — Entropy-based grouping like BLT
- **IntegralLNN** — Context accumulation via integration (our innovation)
- **Sparse Attention** — O(n log n) dilated attention
- **Efficient inference** — Linear-time patching + subquadratic attention

## Why Tokenizer-Free?

| Problem with Tokenizers | Our Solution |
|------------------------|--------------|
| Can't count letters ("r in strawberry") | Direct byte access |
| Arithmetic failures (multi-token numbers) | Each digit = 1 byte |
| Multilingual inequality | All languages = bytes |
| Glitch tokens | No vocabulary = no glitches |
| Fixed compute per token | Dynamic via entropy patching |

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         ionLlama v12                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input: Raw UTF-8 bytes [0-255]                                │
│         "Hello" → [72, 101, 108, 108, 111]                     │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              BYTE EMBEDDING + POSITION                   │   │
│  │  • 256 learnable byte embeddings                        │   │
│  │  • N-gram embedding (context from previous bytes)       │   │
│  │  • Rotary Position Embeddings (RoPE)                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               ENTROPY ESTIMATOR (Small)                  │   │
│  │  • 2-layer Mamba SSM (~1M params)                       │   │
│  │  • Predicts next-byte entropy at each position          │   │
│  │  • O(n) linear time                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │               DYNAMIC PATCHER                            │   │
│  │  • Start new patch when entropy > threshold             │   │
│  │  • Or when patch reaches max_size (e.g., 8 bytes)       │   │
│  │  • Pool bytes → patch representations                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│        Patches: Variable-length semantic units                  │
│        "Hello world" → ["Hello", " wor", "ld"]                 │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LOCAL ENCODER (IntegralLNN)                 │   │
│  │  • Our innovation: h = ∫f(h,x)dt                        │   │
│  │  • Conv1D-based for efficiency                          │   │
│  │  • Accumulates context within each patch                │   │
│  │  • 2-4 layers, lightweight                              │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           LATENT TRANSFORMER (Main Model)                │   │
│  │  • Sparse Dilated Attention: O(n log n)                 │   │
│  │  • Each patch attends to: [i, i-1, i-2, i-4, i-8, ...]  │   │
│  │  • SwiGLU FFN + RMSNorm                                 │   │
│  │  • This is where most parameters live                   │   │
│  │  • 12-32 layers depending on scale                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              LOCAL DECODER (IntegralLNN)                 │   │
│  │  • Cross-attention: patches → bytes                     │   │
│  │  • Unpatch back to byte-level representations           │   │
│  │  • 2-4 layers, mirrors encoder                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              BYTE PREDICTION HEAD                        │   │
│  │  • Linear projection to 256 logits                      │   │
│  │  • Cross-entropy loss per byte                          │   │
│  │  • Optional: contrastive loss (embedding space)         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                           ↓                                     │
│  Output: Next byte probabilities [256]                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Innovations

### 1. IntegralLNN (Our Contribution)
Standard LNN: `dh/dt = -h/τ + f(h, x)` (differential)
IntegralLNN: `h = ∫f(h, x)dt` (integral)

**Why integration > differentiation for language:**
- Language ACCUMULATES meaning over context
- Differentiation measures instantaneous change (loses context)
- Integration preserves and accumulates information

**Implementation:** Conv1D with cumulative sum
```python
def integral_lnn(x):
    # Nonlinear transform
    gate = sigmoid(conv1d_gate(x))
    candidate = tanh(conv1d_candidate(x))
    
    # Integration via cumsum (efficient!)
    integrated = torch.cumsum(gate * candidate, dim=1)
    
    # Decay factor to prevent explosion
    decay = torch.exp(-alpha * position_indices)
    
    return integrated * decay
```

### 2. Entropy-Based Dynamic Patching
Small Mamba model estimates: "How hard is the next byte to predict?"

```python
def compute_patches(bytes, entropy_model, threshold=2.0, max_patch=8):
    # Get entropy at each position
    entropies = entropy_model(bytes)  # O(n)
    
    patches = []
    current_patch = []
    
    for i, (byte, entropy) in enumerate(zip(bytes, entropies)):
        current_patch.append(byte)
        
        # Start new patch if:
        # 1. Entropy spike (hard to predict)
        # 2. Patch reached max size
        if entropy > threshold or len(current_patch) >= max_patch:
            patches.append(current_patch)
            current_patch = []
    
    return patches
```

### 3. O(n log n) Sparse Dilated Attention
Each position attends to exponentially-spaced previous positions:

```
Position 100 attends to: [100, 99, 98, 96, 92, 84, 68, 36, ...]
                              -1  -2  -4  -8  -16 -32 -64

Only ~log(n) positions per token instead of n!
```

**Complexity:**
- Standard attention: O(n²)
- Sparse dilated: O(n log n)
- For 32K sequence: 2,000× faster

### 4. Hybrid Architecture
Combine the best of each approach:

| Component | Architecture | Complexity | Purpose |
|-----------|-------------|------------|---------|
| Entropy Model | Mamba SSM | O(n) | Fast patching |
| Local Encoder | IntegralLNN | O(n) | Byte-level context |
| Latent Transformer | Sparse Attention | O(n log n) | Cross-patch reasoning |
| Local Decoder | IntegralLNN | O(n) | Byte reconstruction |

**Total: O(n log n)** vs BLT's O(n²) on patches

## Parameter Budget

### ionLlama-v12-Small (50M)
```
Entropy Model:    1M   (2-layer Mamba)
Byte Embeddings:  0.1M (256 × 384)
Local Encoder:    5M   (2 IntegralLNN layers)
Latent Transformer: 40M (8 layers, d=512)
Local Decoder:    3M   (2 IntegralLNN layers)
Prediction Head:  0.1M (384 → 256)
```

### ionLlama-v12-Base (200M)
```
Entropy Model:    2M
Byte Embeddings:  0.2M (256 × 768)
Local Encoder:    15M  (4 IntegralLNN layers)
Latent Transformer: 170M (16 layers, d=1024)
Local Decoder:    10M  (4 IntegralLNN layers)
Prediction Head:  0.2M
```

### ionLlama-v12-Large (1B)
```
Entropy Model:    5M
Byte Embeddings:  0.5M (256 × 2048)
Local Encoder:    50M  (6 IntegralLNN layers)
Latent Transformer: 900M (32 layers, d=2048)
Local Decoder:    40M  (6 IntegralLNN layers)
Prediction Head:  0.5M
```

## Training Strategy

### Phase 1: Entropy Model Pre-training
Train small Mamba to predict next byte:
```
Input: bytes[0:i]
Target: bytes[i]
Loss: Cross-entropy
```
This gives us the entropy estimator for patching.

### Phase 2: End-to-End Training
Train full model with frozen entropy model:
```
Input: byte sequence
Target: next byte at each position
Loss: Cross-entropy (byte-level)
```

### Phase 3: Joint Fine-tuning
Unfreeze entropy model, fine-tune everything:
- Entropy model learns task-optimal patching
- Main model adapts to patching decisions

## Efficiency Analysis

### Sequence Length Comparison
For "Hello, how are you today?" (28 characters):

| Method | Tokens/Patches | Attention Cost |
|--------|---------------|----------------|
| GPT-4 BPE | 7 tokens | 7² = 49 |
| Byte-level | 28 bytes | 28² = 784 |
| BLT (patches) | ~7 patches | 7² = 49 |
| **ionLlama v12** | ~7 patches | 7 × log(7) ≈ 20 |

### Memory Usage
- No 100K vocabulary embedding matrix
- Just 256 byte embeddings
- Saves ~400M parameters at GPT-4 scale!

## Advantages Over BLT

| Aspect | BLT (Meta) | ionLlama v12 |
|--------|-----------|--------------|
| Attention | O(n²) on patches | O(n log n) sparse |
| Context model | Standard transformer | IntegralLNN (accumulation) |
| Entropy estimation | Learned CNN | Mamba SSM (linear time) |
| Total complexity | O(n²) | O(n log n) |

## Evaluation Plan

### Benchmarks
1. **Bits-per-Byte (BPB)** — Standard language modeling metric
2. **Character-level tasks:**
   - Letter counting ("How many r's in strawberry?")
   - String reversal
   - Spelling tasks
3. **Arithmetic** — Multi-digit addition/multiplication
4. **Multilingual** — Fair comparison across languages
5. **Robustness** — Typos, noise, adversarial inputs

### Baselines
- LLaMA 3 8B (tokenized)
- BLT 8B (byte-level, standard attention)
- MambaByte (byte-level, SSM)
- GPT-2 (tokenized, classic)

## File Structure

```
ionllama/v12/
├── ARCHITECTURE.md          # This file
├── model.py                 # Full model implementation
├── entropy_model.py         # Mamba-based entropy estimator
├── integral_lnn.py          # IntegralLNN layers
├── sparse_attention.py      # O(n log n) dilated attention
├── patcher.py               # Dynamic byte→patch conversion
├── train.py                 # Training script
├── eval.py                  # Evaluation benchmarks
└── configs/
    ├── small.yaml           # 50M config
    ├── base.yaml            # 200M config
    └── large.yaml           # 1B config
```

## Next Steps

1. Implement entropy_model.py (Mamba SSM)
2. Implement integral_lnn.py (Conv1D-based)
3. Implement sparse_attention.py (dilated pattern)
4. Implement patcher.py (entropy-based grouping)
5. Assemble model.py (full architecture)
6. Create train.py (byte-level training)
7. Benchmark against baselines

---

*ionLlama v12: The first tokenizer-free model with O(n log n) complexity.*
*Designed: 2026-02-28*
