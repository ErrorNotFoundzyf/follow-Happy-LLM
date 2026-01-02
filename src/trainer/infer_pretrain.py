import argparse
import os
import torch
from transformers import AutoTokenizer

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.llama2.config import ModelConfig
from src.llama2.model import Transformer


def load_model(ckpt_path: str, tokenizer_dir: str, dim: int, n_layers: int, vocab_size: int, max_seq_len: int, device: str):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    cfg = ModelConfig(
        dim=dim,
        n_layers=n_layers,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
    model = Transformer(cfg)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Pretrain inference")
    parser.add_argument("--ckpt", type=str, required=True, help="路径: pretrain_*.pth")
    parser.add_argument("--tokenizer_dir", type=str, default="src/tokenization", help="tokenizer 目录")
    parser.add_argument("--prompt", type=str, required=True, help="用户输入提示语")
    parser.add_argument("--device", type=str, default="cuda:0", help="推理设备")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    # 模型超参需与训练时一致
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--n_layers", type=int, default=18)
    parser.add_argument("--vocab_size", type=int, default=6144)
    parser.add_argument("--max_seq_len", type=int, default=512)

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    model, tokenizer = load_model(
        ckpt_path=args.ckpt,
        tokenizer_dir=args.tokenizer_dir,
        dim=args.dim,
        n_layers=args.n_layers,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len,
        device=device,
    )

    # 编码 prompt
    input_ids = tokenizer(args.prompt).data["input_ids"]
    idx = torch.tensor([input_ids], dtype=torch.long, device=device)

    with torch.no_grad():
        out_ids = model.generate(
            idx,
            stop_id=getattr(tokenizer, "eos_token_id", None),
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )

    # 解码完整序列（包含原始 prompt）
    full_ids = torch.cat([idx, out_ids], dim=1)[0].tolist()
    text = tokenizer.decode(full_ids, skip_special_tokens=True)
    print("\n=== Generated ===\n")
    print(text)


if __name__ == "__main__":
    main()
