#!/bin/bash

# 设置数据路径
DATA_PATH="/root/data-fs/Data/seq_monkey_datawhale.jsonl"

# 设置输出目录
OUT_DIR="/root/data-fs/LLM_model/My_llama_pretrain"

# 运行预训练脚本
# 使用 nohup 后台运行，并将日志输出到 pretrain.log
# 如果你想在前台运行，去掉 nohup 和 & 即可
python src/trainer/pretrain.py \
    --data_path "$DATA_PATH" \
    --out_dir "$OUT_DIR" \
    --batch_size 4 \
    --accumulation_steps 32 \
    --dim 1024 \
    --n_layers 18 \
    --vocab_size 6144 \
    --max_seq_len 512 \
    --epochs 1 \
    --num_workers 0 \
    --log_interval 10 \
    --save_interval 500
