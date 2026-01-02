import os
import argparse
import time
import math
import torch
from torch import optim
from torch.utils.data import DataLoader
from contextlib import nullcontext
from transformers import AutoTokenizer

# 添加项目根目录到 sys.path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.llama2.config import ModelConfig
from src.llama2.model import Transformer
from src.dataset import SFTDataset


def Logger(content):
	print(content)


def get_lr(it, total_steps, args):
	warmup_iters = args.warmup_iters
	min_lr = args.learning_rate / 10

	if it < warmup_iters:
		return args.learning_rate * it / warmup_iters if warmup_iters > 0 else args.learning_rate

	if it >= total_steps:
		return min_lr

	decay_ratio = (it - warmup_iters) / max(total_steps - warmup_iters, 1)
	decay_ratio = max(0.0, min(1.0, decay_ratio))
	coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
	return min_lr + coeff * (args.learning_rate - min_lr)


def save_checkpoint(model, args, suffix=""):
	os.makedirs(args.out_dir, exist_ok=True)
	name = f"sft_{args.dim}_{args.n_layers}_{args.vocab_size}{suffix}.pth"
	path = os.path.join(args.out_dir, name)
	state_dict = model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict()
	torch.save(state_dict, path)
	Logger(f"Saved checkpoint to {path}")


def init_model(args):
	tokenizer_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokenization"))
	Logger(f"Loading tokenizer from {tokenizer_path}")
	tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

	lm_config = ModelConfig(
		dim=args.dim,
		n_layers=args.n_layers,
		vocab_size=args.vocab_size,
		max_seq_len=args.max_seq_len
	)

	model = Transformer(lm_config)

	if args.load_ckpt and os.path.isfile(args.load_ckpt):
		state = torch.load(args.load_ckpt, map_location="cpu")
		model.load_state_dict(state, strict=False)
		Logger(f"Loaded checkpoint from {args.load_ckpt}")

	if torch.cuda.device_count() > 1 and "cuda" in args.device:
		Logger(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
		model = torch.nn.DataParallel(model)

	model = model.to(args.device)
	return model, tokenizer


def train_epoch(epoch, args, model, tokenizer, train_loader, optimizer, scaler, ctx, total_steps):
	start_time = time.time()
	iter_per_epoch = len(train_loader)

	for step, (X, Y, loss_mask) in enumerate(train_loader):
		global_step = epoch * iter_per_epoch + step
		X = X.to(args.device)
		Y = Y.to(args.device)
		loss_mask = loss_mask.to(args.device)

		lr = get_lr(global_step, total_steps, args)
		for pg in optimizer.param_groups:
			pg['lr'] = lr

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
			eta_min = (spend_time / (step + 1)) * (iter_per_epoch - step - 1) / 60
			Logger(
				f"Epoch:[{epoch + 1}/{args.epochs}]({step}/{iter_per_epoch}) "
				f"loss:{loss.item() * args.accumulation_steps:.3f} lr:{optimizer.param_groups[-1]['lr']:.7f} "
				f"ETA:{eta_min:.1f}min")

		if (step + 1) % args.save_interval == 0:
			model.eval()
			save_checkpoint(model, args, suffix=f"_step{global_step+1}")
			model.train()


def main():
	parser = argparse.ArgumentParser(description="Tiny-LLM SFT")
	parser.add_argument("--out_dir", type=str, default="/root/data-fs/LLM_model/My_llama_sft", help="模型输出目录")
	parser.add_argument("--data_path", type=str, default="/root/data-fs/Data/sft_data.jsonl", help="SFT数据路径")
	parser.add_argument("--load_ckpt", type=str, default="", help="预训练权重路径，可为空")
	parser.add_argument("--epochs", type=int, default=1, help="训练轮数")
	parser.add_argument("--batch_size", type=int, default=32, help="微批大小")
	parser.add_argument("--accumulation_steps", type=int, default=4, help="梯度累积步数")
	parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率")
	parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="训练设备")
	parser.add_argument("--dtype", type=str, default="bfloat16", help="数据类型")
	parser.add_argument("--num_workers", type=int, default=8, help="DataLoader worker 数")
	parser.add_argument("--grad_clip", type=float, default=1.0, help="梯度裁剪阈值")
	parser.add_argument("--warmup_iters", type=int, default=0, help="学习率预热步数")
	parser.add_argument("--log_interval", type=int, default=50, help="日志间隔")
	parser.add_argument("--save_interval", type=int, default=20000, help="保存间隔（按step），调大可减少文件数量")
	# 模型参数
	parser.add_argument("--dim", type=int, default=1024, help="模型维度")
	parser.add_argument("--n_layers", type=int, default=18, help="层数")
	parser.add_argument("--vocab_size", type=int, default=6144, help="词表大小")
	parser.add_argument("--max_seq_len", type=int, default=512, help="最大序列长度")

	args = parser.parse_args()

	os.makedirs(args.out_dir, exist_ok=True)
	torch.manual_seed(42)

	device_type = "cuda" if "cuda" in args.device else "cpu"
	ctx = nullcontext() if device_type == "cpu" else torch.cuda.amp.autocast()

	model, tokenizer = init_model(args)

	train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
	train_loader = DataLoader(
		train_ds,
		batch_size=args.batch_size,
		pin_memory=True,
		drop_last=False,
		shuffle=True,
		num_workers=args.num_workers,
		persistent_workers=args.num_workers > 0,
	)

	total_steps = len(train_loader) * args.epochs
	scaler = torch.cuda.amp.GradScaler(enabled=(args.dtype in ["float16", "bfloat16"]))
	optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)

	for epoch in range(args.epochs):
		train_epoch(epoch, args, model, tokenizer, train_loader, optimizer, scaler, ctx, total_steps)

	save_checkpoint(model, args, suffix="_final")


if __name__ == "__main__":
	main()
