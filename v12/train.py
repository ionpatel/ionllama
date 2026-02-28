#!/usr/bin/env python3
"""
ionLlama v12 Training Script

Train the tokenizer-free byte-level model on text data.

Usage:
    python train.py --size small --data_path data/train.txt
    python train.py --size base --data_path data/ --wandb
    
For cloud (A100/H100/B200):
    python train.py --size large --data_path tinystories --batch_size 32
"""

import os
import sys
import math
import time
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from model import ionLlamaV12, create_model


class ByteDataset(Dataset):
    """
    Dataset that loads text files and converts to bytes.
    
    Creates (input, target) pairs where target is shifted by 1.
    """
    
    def __init__(
        self,
        data_path: str,
        seq_len: int = 512,
        stride: int = 256,
    ):
        self.seq_len = seq_len
        self.stride = stride
        
        # Load data
        data_path = Path(data_path)
        if data_path.is_file():
            with open(data_path, 'rb') as f:
                self.data = f.read()
        elif data_path.is_dir():
            # Concatenate all .txt files
            data = b""
            for txt_file in sorted(data_path.glob("**/*.txt")):
                with open(txt_file, 'rb') as f:
                    data += f.read() + b"\n"
            self.data = data
        else:
            raise ValueError(f"Data path not found: {data_path}")
        
        print(f"Loaded {len(self.data):,} bytes ({len(self.data)/1e6:.1f} MB)")
        
        # Precompute number of samples
        self.n_samples = max(1, (len(self.data) - seq_len - 1) // stride)
        print(f"Number of samples: {self.n_samples:,}")
        
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = idx * self.stride
        end = start + self.seq_len + 1  # +1 for target
        
        # Get bytes
        chunk = self.data[start:end]
        
        # Convert to tensor
        bytes_tensor = torch.tensor(list(chunk), dtype=torch.long)
        
        # Split input/target
        input_bytes = bytes_tensor[:-1]
        target_bytes = bytes_tensor[1:]
        
        return {
            'input': input_bytes,
            'target': target_bytes,
        }


class Trainer:
    """Training loop for ionLlama v12."""
    
    def __init__(
        self,
        model: ionLlamaV12,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        warmup_steps: int = 100,
        max_steps: int = 10000,
        grad_clip: float = 1.0,
        log_interval: int = 10,
        eval_interval: int = 500,
        save_interval: int = 1000,
        save_dir: str = "checkpoints",
        device: str = "auto",
        use_wandb: bool = False,
        run_name: str = "ionllama_v12",
    ):
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        print(f"Using device: {self.device}")
        
        # Model
        self.model = model.to(self.device)
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.95),
        )
        
        # Scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max_steps,
            eta_min=lr * 0.1,
        )
        
        # Training params
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.grad_clip = grad_clip
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
                project="ionllama",
                name=run_name,
                config={
                    "model_size": sum(p.numel() for p in model.parameters()),
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "max_steps": max_steps,
                }
            )
        
        # State
        self.step = 0
        self.best_val_loss = float('inf')
        
    def _get_lr(self) -> float:
        """Get learning rate with warmup."""
        if self.step < self.warmup_steps:
            return self.optimizer.param_groups[0]['lr'] * (self.step / self.warmup_steps)
        return self.scheduler.get_last_lr()[0]
    
    def _set_lr(self, lr: float):
        """Set learning rate."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation on validation set."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0
        total_bpb = 0
        total_patches = 0
        total_bytes = 0
        n_batches = 0
        
        for batch in self.val_loader:
            input_bytes = batch['input'].to(self.device)
            target_bytes = batch['target'].to(self.device)
            
            output = self.model(input_bytes, target_bytes)
            
            total_loss += output['loss'].item()
            total_bpb += output['bpb'].item()
            total_patches += output['n_patches'].sum().item()
            total_bytes += input_bytes.numel()
            n_batches += 1
            
            if n_batches >= 50:  # Limit eval batches
                break
        
        self.model.train()
        
        metrics = {
            'val_loss': total_loss / n_batches,
            'val_bpb': total_bpb / n_batches,
            'val_compression': total_bytes / total_patches if total_patches > 0 else 1.0,
        }
        
        return metrics
    
    def save_checkpoint(self, name: str = "latest"):
        """Save model checkpoint."""
        checkpoint = {
            'step': self.step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }
        
        path = self.save_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.step = checkpoint['step']
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        print(f"Loaded checkpoint from {path} (step {self.step})")
    
    def train(self):
        """Main training loop."""
        self.model.train()
        
        # Metrics tracking
        running_loss = 0
        running_bpb = 0
        running_patches = 0
        running_bytes = 0
        start_time = time.time()
        
        # Infinite dataloader
        data_iter = iter(self.train_loader)
        
        while self.step < self.max_steps:
            # Get batch
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                batch = next(data_iter)
            
            input_bytes = batch['input'].to(self.device)
            target_bytes = batch['target'].to(self.device)
            
            # Forward
            output = self.model(input_bytes, target_bytes)
            loss = output['loss']
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.grad_clip
                )
            else:
                grad_norm = 0
            
            # Update weights
            lr = self._get_lr()
            self._set_lr(lr)
            self.optimizer.step()
            
            if self.step >= self.warmup_steps:
                self.scheduler.step()
            
            # Track metrics
            running_loss += loss.item()
            running_bpb += output['bpb'].item()
            running_patches += output['n_patches'].sum().item()
            running_bytes += input_bytes.numel()
            
            self.step += 1
            
            # Logging
            if self.step % self.log_interval == 0:
                elapsed = time.time() - start_time
                
                avg_loss = running_loss / self.log_interval
                avg_bpb = running_bpb / self.log_interval
                compression = running_bytes / running_patches if running_patches > 0 else 1.0
                throughput = running_bytes / elapsed
                
                print(f"Step {self.step:6d} | "
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
                        'train/grad_norm': grad_norm if isinstance(grad_norm, float) else grad_norm.item(),
                    }, step=self.step)
                
                # Reset
                running_loss = 0
                running_bpb = 0
                running_patches = 0
                running_bytes = 0
                start_time = time.time()
            
            # Evaluation
            if self.step % self.eval_interval == 0:
                val_metrics = self.evaluate()
                
                if val_metrics:
                    print(f"  → Val Loss {val_metrics['val_loss']:.4f} | "
                          f"Val BPB {val_metrics['val_bpb']:.3f}")
                    
                    if self.use_wandb:
                        import wandb
                        wandb.log(val_metrics, step=self.step)
                    
                    # Save best
                    if val_metrics['val_loss'] < self.best_val_loss:
                        self.best_val_loss = val_metrics['val_loss']
                        self.save_checkpoint("best")
            
            # Save checkpoint
            if self.step % self.save_interval == 0:
                self.save_checkpoint("latest")
        
        # Final save
        self.save_checkpoint("final")
        print(f"\nTraining complete! Best val loss: {self.best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train ionLlama v12")
    
    # Model
    parser.add_argument("--size", type=str, default="small",
                        choices=["tiny", "small", "base", "large", "xl"])
    
    # Data
    parser.add_argument("--data_path", type=str, default="data/train.txt")
    parser.add_argument("--val_path", type=str, default=None)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--stride", type=int, default=256)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    
    # Logging
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--run_name", type=str, default="ionllama_v12")
    
    # Device
    parser.add_argument("--device", type=str, default="auto")
    
    # Resume
    parser.add_argument("--resume", type=str, default=None)
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("ionLlama v12: Tokenizer-Free Training")
    print("=" * 60)
    
    # Create model
    print(f"\nCreating {args.size} model...")
    model = create_model(args.size)
    
    param_counts = model.count_parameters()
    print(f"Total parameters: {param_counts['total']:,}")
    
    # Create datasets
    print(f"\nLoading data from {args.data_path}...")
    train_dataset = ByteDataset(
        args.data_path,
        seq_len=args.seq_len,
        stride=args.stride,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    
    val_loader = None
    if args.val_path:
        val_dataset = ByteDataset(
            args.val_path,
            seq_len=args.seq_len,
            stride=args.seq_len,  # No overlap for val
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
        )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        log_interval=args.log_interval,
        eval_interval=args.eval_interval,
        save_interval=args.save_interval,
        save_dir=args.save_dir,
        device=args.device,
        use_wandb=args.wandb,
        run_name=args.run_name,
    )
    
    # Resume if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\nStarting training...")
    trainer.train()


if __name__ == "__main__":
    main()
