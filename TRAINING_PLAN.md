# ionLlama v12 Training Plan

## 🎯 Goal
Train and benchmark ionLlama v12 against top models to validate:
1. Tokenizer-free approach works at scale
2. O(n log n) attention matches quality of O(n²)
3. IntegralLNN improves over standard transformer
4. Character-level tasks (spelling, arithmetic) improved

---

## 📊 Experiment Matrix

### Phase 1: Architecture Validation (TinyStories)
| Model | Params | Dataset | Steps | Expected BPB | GPU Hours | Cost |
|-------|--------|---------|-------|--------------|-----------|------|
| ionLlama-tiny | 5M | TinyStories | 10K | <2.0 | 0.5 | $2.50 |
| ionLlama-small | 50M | TinyStories | 30K | <1.5 | 2 | $10 |
| ionLlama-base | 200M | TinyStories | 50K | <1.2 | 8 | $40 |

### Phase 2: Scale Validation (FineWeb-Edu)
| Model | Params | Dataset | Steps | Expected BPB | GPU Hours | Cost |
|-------|--------|---------|-------|--------------|-----------|------|
| ionLlama-base | 200M | FineWeb | 100K | <0.9 | 24 | $120 |
| ionLlama-large | 800M | FineWeb | 100K | <0.75 | 72 | $360 |

### Phase 3: Benchmark vs Baselines
Compare against:
- GPT-2 124M (tokenized baseline)
- BLT 8B (byte-level, reported numbers)
- MambaByte 353M (byte-level SSM)
- LLaMA 3 8B (tokenized SOTA)

---

## 🚀 RunPod Setup

### Recommended Instance
- **GPU:** A100 80GB or H100 80GB
- **Cost:** $2-5/hr
- **Storage:** 50GB minimum

### One-Liner Setup
```bash
# SSH into RunPod, then:
git clone https://github.com/ionpatel/ionllama.git
cd ionllama/v12
pip install torch datasets wandb tqdm huggingface_hub
python train_cloud.py --size small --dataset tinystories --max_steps 30000 --wandb
```

---

## 📋 Training Commands

### Quick Test (5 min)
```bash
python train_cloud.py --size tiny --dataset tinystories --max_steps 1000
```

### Small Model Full Training
```bash
python train_cloud.py \
    --size small \
    --dataset tinystories \
    --max_steps 50000 \
    --batch_size 16 \
    --grad_accum 4 \
    --wandb \
    --run_name "ionllama-small-tinystories"
```

### Base Model Training
```bash
python train_cloud.py \
    --size base \
    --dataset fineweb \
    --max_steps 100000 \
    --batch_size 32 \
    --grad_accum 8 \
    --lr 2e-4 \
    --wandb \
    --benchmark_after
```

### Large Model Training (H100 recommended)
```bash
python train_cloud.py \
    --size large \
    --dataset fineweb \
    --max_steps 200000 \
    --batch_size 64 \
    --grad_accum 8 \
    --mixed_precision bf16 \
    --wandb
```

---

## 📈 Benchmarks to Run

After training, run comprehensive benchmarks:

```bash
python benchmark.py --size small --checkpoint checkpoints/best.pt --output results.json
```

### Metrics Tracked
1. **Language Modeling**
   - Bits-per-Byte (BPB) ↓
   - Perplexity (PPL) ↓

2. **Character Tasks** (where we should excel!)
   - Letter counting accuracy ("r in strawberry")
   - Spelling accuracy
   - String reversal
   
3. **Arithmetic**
   - Addition accuracy (multi-digit)
   - Multiplication accuracy

4. **Speed**
   - Throughput (bytes/sec)
   - Latency (ms)
   - Generation speed

5. **Memory**
   - Peak VRAM

6. **Compression**
   - Bytes → patches ratio

7. **Multilingual Fairness**
   - BPB across: English, Chinese, Arabic, Code

---

## 🎯 Success Criteria

### Minimum Viable Success
- [ ] BPB < 1.5 on TinyStories (match GPT-2 level)
- [ ] Training stable for 50K+ steps
- [ ] Generation produces coherent text

### Target Success
- [ ] BPB < 1.0 on TinyStories (beat GPT-2)
- [ ] Letter counting >80% (vs GPT-2's ~10%)
- [ ] Arithmetic addition >50% (vs GPT-2's ~30%)
- [ ] Throughput >50KB/s on A100

### Stretch Goals
- [ ] BPB < 0.7 (match BLT/LLaMA 3)
- [ ] Compression ratio >3x (entropy patching working)
- [ ] Publish results

---

## 💰 Budget Estimate

| Phase | GPU | Hours | Cost/hr | Total |
|-------|-----|-------|---------|-------|
| Phase 1 | A100 | 10 | $2 | $20 |
| Phase 2 | A100 | 100 | $2 | $200 |
| Phase 3 | A100 | 20 | $2 | $40 |
| **Total** | | **130** | | **$260** |

*Harshil had $88 RunPod credits + $300 budget from before*

---

## 📁 Files to Watch

```
ionllama/v12/
├── checkpoints/
│   ├── best.pt          # Best validation checkpoint
│   ├── latest.pt        # Most recent checkpoint
│   └── step_*.pt        # Periodic saves
├── benchmark_results.json   # Benchmark output
└── wandb/               # Training logs
```

---

## 🔬 After Training

1. **Download checkpoints** to Mac:
   ```bash
   scp runpod:/workspace/ionllama/v12/checkpoints/best.pt ~/ionllama-weights/
   ```

2. **Run full benchmarks locally**

3. **Compare vs baselines** using benchmark.py

4. **Write up results** for potential publication

---

## 📝 Notes

- Start with tiny/small to validate before scaling
- Monitor W&B for training curves
- Check compression ratio - if stuck at 1x, entropy model needs more training
- Character tasks require trained model - don't benchmark untrained

---

*Ready to train! Start with Phase 1 to validate, then scale up.*
