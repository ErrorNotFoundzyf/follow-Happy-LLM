#!/usr/bin/env python3
# train_tokenizer.py
import sys
from pathlib import Path

# 将项目根目录添加到Python路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入你的tokenizer训练函数
from src.tokenization.tokenizer import train_tokenizer, quick_test

def main():
    # 配置参数
    DATA_PATH = "/root/data-fs/Data/seq_monkey_datawhale.jsonl"
    SAVE_DIR = "/root/Happy-llm/happy-llm-project/src/tokenization"
    VOCAB_SIZE = 6144  # 与 pretrain.py 默认参数保持一致
    BATCH_SIZE = 5000  # 缩小批量以控制内存峰值
    MAX_LINES = 1_000_000  # 抽样100万行，进一步降低内存占用
    
    print("=" * 60)
    print("1. 快速数据测试")
    print("=" * 60)
    
    # 先进行快速测试
    quick_test(DATA_PATH, test_lines=20000)
    
    print("\n" + "=" * 60)
    print("2. 开始训练Tokenizer（测试模式）")
    print("=" * 60)
    print(f"训练数据: {DATA_PATH}")
    print(f"输出目录: {SAVE_DIR}")
    print(f"词汇表大小: {VOCAB_SIZE}")
    print(f"批处理大小: {BATCH_SIZE}")
    print(f"最大行数: {MAX_LINES:,}")
    print("=" * 60)
    
    # 开始训练
    train_tokenizer(
        data_path=DATA_PATH,
        save_dir=SAVE_DIR,
        vocab_size=VOCAB_SIZE,
        batch_size=BATCH_SIZE,
        test_mode=False,  # 全量训练
        max_lines=MAX_LINES
    )
    
    print("\n" + "=" * 60)
    print("✅ 测试训练完成!")
    print(f"请检查目录: {SAVE_DIR}")
    print("\n如果测试成功，可以修改 train_tokenizer.py")
    print("将 test_mode=False 进行完整训练")
    print("=" * 60)

if __name__ == "__main__":
    main()