import os
import platform
import argparse
import time
import math
import warnings
import torch
import torch.distributed as dist
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer

# 添加项目根目录到 sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.llama2.config import ModelConfig
from src.llama2.model import Transformer
from src.dataset import PretrainDataset

import swanlab

# 忽略警告
warnings.filterwarnings('ignore')

def Logger(content):
    print(content)

def get_lr(it, all, args):
    warmup_iters = args.warmup_iters
    lr_decay_iters = all
    min_lr = args.learning_rate / 10

    if it < warmup_iters:
        return args.learning_rate * it / warmup_iters
    
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (args.learning_rate - min_lr)

def train_epoch(epoch, args, model, tokenizer, train_loader, optimizer, scaler, ctx):
    start_time = time.time()
    iter_per_epoch = len(train_loader)
    
    for step, (X, Y, loss_mask) in enumerate(train_loader):
        if args.max_steps is not None and step >= args.max_steps:
            Logger(f"Reached max_steps {args.max_steps}, stopping epoch.")
            break

        X = X.to(args.device)
        Y = Y.to(args.device)
        loss_mask = loss_mask.to(args.device)

        lr = get_lr(epoch * iter_per_epoch + step, args.epochs * iter_per_epoch, args)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        with ctx:
            out = model(X, Y)
            loss = out.last_loss / args.accumulation_steps
            loss_mask = loss_mask.view(-1)
            loss = torch.sum(loss * loss_mask) / loss_mask.sum()

        scaler.scale(loss).backward()

        if (step + 1) % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            optimizer.zero_grad(set_to_none=True)

        if step % args.log_interval == 0:
            spend_time = time.time() - start_time
            Logger(
                'Epoch:[{}/{}]({}/{}) loss:{:.3f} lr:{:.7f} epoch_Time:{}min;'.format(
                    epoch + 1,
                    args.epochs,
                    step,
                    iter_per_epoch,
                    loss.item() * args.accumulation_steps,
                    optimizer.param_groups[-1]['lr'],
                    spend_time / (step + 1) * iter_per_epoch // 60 - spend_time // 60))
            
            if args.use_swanlab:
                swanlab.log({
                    "loss": loss.item() * args.accumulation_steps,
                    "lr": optimizer.param_groups[-1]['lr']
                })

        if (step + 1) % args.save_interval == 0:
            model.eval()
            # 确保保存目录存在
            os.makedirs(args.save_dir, exist_ok=True)
            ckp = f'{args.save_dir}/pretrain_{args.dim}_{args.n_layers}_{args.vocab_size}.pth'
            
            if isinstance(model, torch.nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
                
            torch.save(state_dict, ckp)
            Logger(f"Saved checkpoint to {ckp}")
            model.train()

        if (step + 1) % 20000 == 0:
            model.eval()
            os.makedirs(args.save_dir, exist_ok=True)
            ckp = f'{args.save_dir}/pretrain_{args.dim}_{args.n_layers}_{args.vocab_size}_step{step+1}.pth'
            
            if isinstance(model, torch.nn.DataParallel):
                state_dict = model.module.state_dict()
            else:
                state_dict = model.state_dict()
                
            torch.save(state_dict, ckp)
            Logger(f"Saved checkpoint to {ckp}")
            model.train()

def init_model(args):
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 加载tokenizer
    # 假设 tokenizer 在 src/tokenization 目录下
    tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokenization"))
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    lm_config = ModelConfig(
        dim=args.dim,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len
    )
    
    model = Transformer(lm_config)
    
    if torch.cuda.device_count() > 1:
        Logger(f"Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
    
    model = model.to(args.device)
    Logger(f'LLM总参数量：{count_parameters(model) / 1e6:.3f} 百万')
    return model, tokenizer

def main():
    parser = argparse.ArgumentParser(description="Tiny-LLM Pretraining")
    
    parser.add_argument("--out_dir", type=str, default="/root/data-fs/LLM_model/My_llama_pretrain", help="模型输出目录")
    parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=4, help="批次大小") # 默认为4
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="学习率")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
    parser.add_argument("--use_swanlab", action="store_true", help="是否使用SwanLab")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载的工作进程数")
    parser.add_argument("--data_path", type=str, default="/root/data-fs/Data/seq_monkey_datawhale.jsonl", help="训练数据路径")
    parser.add_argument("--accumulation_steps", type=int, default=32, help="梯度累积步数") # 默认为32
    parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
    parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热迭代次数")
    parser.add_argument("--log_interval", type=int, default=100, help="日志记录间隔")
    parser.add_argument("--save_interval", type=int, default=1000, help="模型保存间隔")
    
    # 模型参数
    parser.add_argument("--dim", type=int, default=1024, help="模型维度")
    parser.add_argument("--n_layers", type=int, default=18, help="层数")
    parser.add_argument("--vocab_size", type=int, default=6144, help="词表大小")
    parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")
    parser.add_argument("--load_model", type=str, default=None, help="加载预训练模型路径")
    parser.add_argument("--max_steps", type=int, default=None, help="最大训练步数（用于继续训练时限制步数）")

    args = parser.parse_args()
    
    args.save_dir = args.out_dir
    os.makedirs(args.out_dir, exist_ok=True)
    
    torch.manual_seed(42)
    
    if args.use_swanlab:
        swanlab.init(
            project="Happy-LLM",
            experiment_name="Pretrain",
            config=args,
        )

    device_type = "cuda" if "cuda" in args.device else "cpu"
    ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()
    
    model, tokenizer = init_model(args)

    if args.load_model:
        Logger(f"Loading model from {args.load_model}")
        state_dict = torch.load(args.load_model, map_location=args.device)
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)
    
    train_ds = PretrainDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        pin_memory=False,            # 降低Host内存压力
        drop_last=False,
        shuffle=True,               # 使用索引随机采样
        num_workers=args.num_workers
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ['float16', 'bfloat16']))
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    for epoch in range(args.epochs):
        train_epoch(epoch, args, model, tokenizer, train_loader, optimizer, scaler, ctx)

    # Save final model
    model.eval()
    os.makedirs(args.save_dir, exist_ok=True)
    ckp = f'{args.save_dir}/pretrain_{args.dim}_{args.n_layers}_{args.vocab_size}_final.pth'
    
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
        
    torch.save(state_dict, ckp)
    Logger(f"Saved final checkpoint to {ckp}")

if __name__ == "__main__":
    main()
