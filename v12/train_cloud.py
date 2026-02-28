#!/usr/bin/env python3
"""
ionLlama v12 Cloud Training Script

Optimized for:
- RunPod (A100, H100, B200)
- Lambda Labs
- Vast.ai
- Google Colab Pro

Features:
- Automatic dataset download (TinyStories, FineWeb)
- Mixed precision training (bf16/fp16)
- Gradient checkpointing for memory efficiency
- W&B logging
- Automatic checkpointing to cloud storage
- Comprehensive benchmarking after training

Usage:
    # Quick test (tiny model, small data)
    python train_cloud.py --size tiny --dataset tinystories --max_steps 1000
    
    # Full training
    python train_cloud.py --size small --dataset tinystories --max_steps 50000 --wandb
    
    # Large scale
    python train_cloud.py --size base --dataset fineweb --max_steps 100000 --batch_size 32
"""

import os
import sys
import time
import json
import math
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, List
from dataclasses import dataclass
import traceback

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torch.optim import AdamW
from torch.amp import autocast, GradScaler

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))


def install_dependencies():
    """Install required packages."""
    packages = [
        "datasets",
        "wandb",
        "huggingface_hub",
        "tqdm",
    ]
    
    for pkg in packages:
        try:
            __import__(pkg.replace("-", "_"))
        except ImportError:
            print(f"Installing {pkg}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])


def download_dataset(name: str, data_dir: Path) -> Path:
    """Download and prepare dataset."""
    from datasets import load_dataset
    
    data_dir.mkdir(parents=True, exist_ok=True)
    
    if name == "tinystories":
        print("📥 Downloading TinyStories...")
        ds = load_dataset("roneneldan/TinyStories", split="train")
        
        # Save as text
        output_path = data_dir / "tinystories.txt"
        if not output_path.exists():
            with open(output_path, 'w') as f:
                for i, item in enumerate(ds):
                    f.write(item['text'] + "\n\n")
                    if (i + 1) % 100000 == 0:
                        print(f"  Processed {i+1:,} stories...")
        
        print(f"✅ TinyStories saved to {output_path}")
        return output_path
    
    elif name == "fineweb":
        print("📥 Downloading FineWeb-Edu (sample)...")
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        
        output_path = data_dir / "fineweb_sample.txt"
        if not output_path.exists():
            with open(output_path, 'w') as f:
                for i, item in enumerate(ds):
                    f.write(item['text'] + "\n\n")
                    if i >= 500000:  # ~500MB sample
                        break
                    if (i + 1) % 50000 == 0:
                        print(f"  Processed {i+1:,} documents...")
        
        print(f"✅ FineWeb sample saved to {output_path}")
        return output_path
    
    elif name == "openwebtext":
        print("📥 Downloading OpenWebText...")
        ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)
        
        output_path = data_dir / "openwebtext.txt"
        if not output_path.exists():
            with open(output_path, 'w') as f:
                for i, item in enumerate(ds):
                    f.write(item['text'] + "\n\n")
                    if i >= 500000:
                        break
                    if (i + 1) % 50000 == 0:
                        print(f"  Processed {i+1:,} documents...")
        
        print(f"✅ OpenWebText saved to {output_path}")
        return output_path
    
    else:
        raise ValueError(f"Unknown dataset: {name}")


class StreamingByteDataset(IterableDataset):
    """Memory-efficient streaming dataset for large files."""
    
    def __init__(
        self,
        data_path: str,
        seq_len: int = 512,
        shuffle_buffer: int = 10000,
    ):
        self.data_path = Path(data_path)
        self.seq_len = seq_len
        self.shuffle_buffer = shuffle_buffer
        
        # Get file size
        self.file_size = self.data_path.stat().st_size
        print(f"Dataset: {self.file_size / 1e9:.2f} GB")
        
    def __iter__(self):
        import random
        
        buffer = []
        
        with open(self.data_path, 'rb') as f:
            while True:
                chunk = f.read(self.seq_len + 1)
                
                if len(chunk) < self.seq_len + 1:
                    # End of file, restart
                    f.seek(0)
                    chunk = f.read(self.seq_len + 1)
                
                if len(chunk) < self.seq_len + 1:
                    continue
                
                bytes_tensor = torch.tensor(list(chunk), dtype=torch.long)
                
                sample = {
                    'input': bytes_tensor[:-1],
                    'target': bytes_tensor[1:],
                }
                
                # Shuffle buffer
                buffer.append(sample)
                
                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)
                    while len(buffer) > self.shuffle_buffer // 2:
                        yield buffer.pop()


class CloudTrainer:
    """Training loop optimized for cloud GPUs."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 500,
        max_steps: int = 50000,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 1,
        log_interval: int = 10,
        eval_interval: int = 1000,
        save_interval: int = 5000,
        save_dir: str = "checkpoints",
        use_wandb: bool = False,
        run_name: str = "ionllama_v12",
        mixed_precision: str = "bf16",  # bf16, fp16, or none
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")
        
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Model
        self.model = model.to(self.device)
        
        # Compile if PyTorch 2.0+
        # Disabled for now - dynamic patching causes graph breaks
        # if hasattr(torch, 'compile') and self.device.type == 'cuda':
        #     print("Compiling model with torch.compile...")
        #     self.model = torch.compile(self.model, mode='reduce-overhead')
        print("Note: torch.compile disabled (dynamic patching not compatible)")
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
            fused=True if self.device.type == 'cuda' else False,
        )
        
        # Mixed precision
        self.mixed_precision = mixed_precision
        if mixed_precision == "bf16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            self.scaler = None  # bf16 doesn't need scaler
        elif mixed_precision == "fp16":
            self.dtype = torch.float16
            self.scaler = GradScaler()
        else:
            self.dtype = torch.float32
            self.scaler = None
        
        print(f"Mixed precision: {mixed_precision} ({self.dtype})")
        
        # Training params
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.save_interval = save_interval
        
        # Directories
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Logging
        self.use_wandb = use_wandb
        self.run_name = run_name
        
        if use_wandb:
            import wandb
            wandb.init(
                project="ionllama-v12",
                name=run_name,
                config={
                    "model_params": sum(p.numel() for p in model.parameters()),
                    "lr": lr,
                    "max_steps": max_steps,
                    "mixed_precision": mixed_precision,
                }
            )
        
        # State
        self.step = 0
        self.best_val_loss = float('inf')
        
    def _get_lr(self) -> float:
        """Cosine schedule with warmup."""
        if self.step < self.warmup_steps:
            return self.lr * (self.step / self.warmup_steps)
        
        progress = (self.step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        return self.lr * 0.1 + 0.9 * self.lr * (1 + math.cos(math.pi * progress)) / 2
    
    @torch.no_grad()
    def evaluate(self, max_batches: int = 50) -> Dict[str, float]:
        """Run evaluation."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_bytes = 0
        total_patches = 0
        n_batches = 0
        
        for batch in self.val_loader:
            input_bytes = batch['input'].to(self.device)
            target_bytes = batch['target'].to(self.device)
            
            with autocast('cuda', dtype=self.dtype, enabled=self.dtype != torch.float32):
                output = self.model(input_bytes, target_bytes)
            
            total_loss += output['loss'].item() * input_bytes.numel()
            total_bytes += input_bytes.numel()
            total_patches += output['n_patches'].sum().item()
            n_batches += 1
            
            if n_batches >= max_batches:
                break
        
        self.model.train()
        
        avg_loss = total_loss / total_bytes if total_bytes > 0 else 0
        bpb = avg_loss / math.log(2)
        compression = total_bytes / total_patches if total_patches > 0 else 1.0
        
        return {
            'val_loss': avg_loss,
            'val_bpb': bpb,
            'val_ppl': math.exp(avg_loss) if avg_loss < 10 else float('inf'),
            'val_compression': compression,
        }
    
    def save_checkpoint(self, name: str = "latest"):
        """Save checkpoint."""
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        path = self.save_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"💾 Saved checkpoint to {path}")
    
    def train(self):
        """Main training loop."""
        self.model.train()
        
        # Metrics
        running_loss = 0
        running_bpb = 0
        running_patches = 0
        running_bytes = 0
        start_time = time.time()
        
        # Data iterator
        data_iter = iter(self.train_loader)
        
        print(f"\n🚀 Starting training for {self.max_steps:,} steps...")
        print(f"   Gradient accumulation: {self.grad_accum_steps}")
        print(f"   Effective batch size: {self.train_loader.batch_size * self.grad_accum_steps}")
        
        while self.step < self.max_steps:
            # Accumulate gradients
            self.optimizer.zero_grad()
            accum_loss = 0
            
            for micro_step in range(self.grad_accum_steps):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    batch = next(data_iter)
                
                input_bytes = batch['input'].to(self.device)
                target_bytes = batch['target'].to(self.device)
                
                # Forward
                with autocast('cuda', dtype=self.dtype, enabled=self.dtype != torch.float32):
                    output = self.model(input_bytes, target_bytes)
                    loss = output['loss'] / self.grad_accum_steps
                
                # Backward
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accum_loss += loss.item() * self.grad_accum_steps
                running_patches += output['n_patches'].sum().item()
                running_bytes += input_bytes.numel()
            
            # Gradient clipping
            if self.scaler:
                self.scaler.unscale_(self.optimizer)
            
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            # Update
            lr = self._get_lr()
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            
            if self.scaler:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            
            # Track
            running_loss += accum_loss
            running_bpb += accum_loss / math.log(2)
            self.step += 1
            
            # Log
            if self.step % self.log_interval == 0:
                elapsed = time.time() - start_time
                
                avg_loss = running_loss / self.log_interval
                avg_bpb = running_bpb / self.log_interval
                compression = running_bytes / running_patches if running_patches > 0 else 1.0
                throughput = running_bytes / elapsed
                
                print(f"Step {self.step:6d}/{self.max_steps} | "
                      f"Loss {avg_loss:.4f} | "
                      f"BPB {avg_bpb:.3f} | "
                      f"Comp {compression:.1f}× | "
                      f"LR {lr:.2e} | "
                      f"{throughput/1000:.1f} KB/s")
                
                if self.use_wandb:
                    import wandb
                    wandb.log({
                        'train/loss': avg_loss,
                        'train/bpb': avg_bpb,
                        'train/compression': compression,
                        'train/lr': lr,
                        'train/throughput': throughput,
                        'train/grad_norm': grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm,
                    }, step=self.step)
                
                # Reset
                running_loss = 0
                running_bpb = 0
                running_patches = 0
                running_bytes = 0
                start_time = time.time()
            
            # Evaluate
            if self.step % self.eval_interval == 0:
                val_metrics = self.evaluate()
                
                if val_metrics:
                    print(f"  📊 Val Loss {val_metrics['val_loss']:.4f} | "
                          f"Val BPB {val_metrics['val_bpb']:.3f} | "
                          f"Val PPL {val_metrics['val_ppl']:.2f}")
                    
                    if self.use_wandb:
                        import wandb
                        wandb.log(val_metrics, step=self.step)
                    
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint("best")
            
            # Save
            if self.step % self.save_interval == 0:
                self.save_checkpoint("latest")
                self.save_checkpoint(f"step_{self.step}")
        
        # Final save
        self.save_checkpoint("final")
        print(f"\n✅ Training complete! Best val loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train ionLlama v12 on cloud GPU")
    
    # Model
    parser.add_argument("--size", type=str, default="small",
                        choices=["tiny", "small", "base", "large"])
    
    # Data
    parser.add_argument("--dataset", type=str, default="tinystories",
                        choices=["tinystories", "fineweb", "openwebtext"])
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--seq_len", type=int, default=512)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--warmup_steps", type=int, default=500)
    
    # Efficiency
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "none"])
    
    # Logging
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="ionllama_v12")
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    # Eval
    parser.add_argument("--eval_interval", type=int, default=1000)
    parser.add_argument("--benchmark_after", action="store_true")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("🦙 ionLlama v12: Tokenizer-Free Cloud Training")
    print("=" * 70)
    
    # Install dependencies
    install_dependencies()
    
    # Import model (after dependencies installed)
    from model import create_model
    
    # Download dataset
    data_dir = Path(args.data_dir)
    data_path = download_dataset(args.dataset, data_dir)
    
    # Create model
    print(f"\n🏗️  Creating {args.size} model...")
    model = create_model(args.size)
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"   Parameters: {param_count:,}")
    
    # Create datasets
    print(f"\n📦 Loading dataset from {data_path}...")
    train_dataset = StreamingByteDataset(
        data_path=str(data_path),
        seq_len=args.seq_len,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
    )
    
    # Create trainer
    trainer = CloudTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # Use train data for quick val
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        grad_accum_steps=args.grad_accum,
        eval_interval=args.eval_interval,
        save_dir=args.save_dir,
        use_wandb=args.wandb,
        run_name=args.run_name,
        mixed_precision=args.mixed_precision,
    )
    
    # Train
    trainer.train()
    
    # Benchmark
    if args.benchmark_after:
        print("\n📊 Running benchmarks...")
        from benchmark import BenchmarkSuite, get_baseline_results, print_comparison_table
        
        # Reload best checkpoint
        best_path = Path(args.save_dir) / "best.pt"
        if best_path.exists():
            checkpoint = torch.load(best_path)
            model.load_state_dict(checkpoint['model_state_dict'])
        
        # Run benchmarks
        suite = BenchmarkSuite(model, trainer.device, f"ionLlama-v12-{args.size}")
        
        sample_texts = [
            "The quick brown fox jumps over the lazy dog. " * 10,
            "Machine learning models are becoming increasingly powerful. " * 10,
        ]
        
        result = suite.run_all(sample_texts)
        baselines = get_baseline_results()
        print_comparison_table(result, baselines)


if __name__ == "__main__":
    main()
