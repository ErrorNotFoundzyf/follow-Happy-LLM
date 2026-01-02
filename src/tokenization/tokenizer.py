import random
import json
import os
import time
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import (
    decoders,
    models,
    pre_tokenizers,
    trainers,
    Tokenizer,
)
from tokenizers.normalizers import NFKC
from typing import Generator, List
import gc

random.seed(42)

def batch_iterator(file_path: str, batch_size: int = 10000, max_lines: int = None) -> Generator[List[str], None, None]:
    """批处理迭代器，避免一次性加载所有数据"""
    batch = []
    line_count = 0
    
    print(f"开始读取文件: {file_path}")
    start_time = time.time()
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if max_lines and line_num > max_lines:
                break
                
            try:
                data = json.loads(line.strip())
                if 'text' not in data:
                    print(f"第 {line_num} 行缺少 'text' 字段")
                    continue
                    
                text = data['text']
                if text and text.strip():  # 确保文本非空
                    batch.append(text)
                    
                # 每批处理 batch_size 条数据
                if len(batch) >= batch_size:
                    line_count += len(batch)
                    elapsed = time.time() - start_time
                    print(f"已处理 {line_count:,} 行，耗时: {elapsed:.1f}s，速度: {line_count/elapsed:.0f} 行/s")
                    yield batch
                    batch = []
                    gc.collect()  # 手动垃圾回收
                    
            except json.JSONDecodeError:
                print(f"第 {line_num} 行 JSON 解析错误")
                continue
            except Exception as e:
                print(f"第 {line_num} 行处理错误: {e}")
                continue
    
    # 返回最后一批
    if batch:
        line_count += len(batch)
        print(f"最后一批 {len(batch)} 行，总计 {line_count:,} 行")
        yield batch
    
    print(f"数据读取完成，总计 {line_count:,} 行，总耗时: {time.time() - start_time:.1f}s")

def create_tokenizer_config(save_dir: str) -> None:
    """创建完整的tokenizer配置文件"""
    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": True,
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "pad_token": "<|im_end|>",
        "unk_token": "<unk>",
        "model_max_length": 8192,  # 改为合理的值
        "clean_up_tokenization_spaces": False,
        "tokenizer_class": "PreTrainedTokenizerFast",
        "chat_template": (
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'user' %}"
            "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
            "{% elif message['role'] == 'assistant' %}"
            "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            "{{ '<|im_start|>assistant\n' }}"
            "{% endif %}"
        )
    }

    # 保存主配置文件
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

    # 创建special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<|im_start|>",
        "eos_token": "<|im_end|>",
        "unk_token": "<unk>",
        "pad_token": "<|im_end|>",
        "additional_special_tokens": ["<s>", "</s>"]
    }
    with open(os.path.join(save_dir, "special_tokens_map.json"), "w", encoding="utf-8") as f:
        json.dump(special_tokens_map, f, ensure_ascii=False, indent=4)

def train_tokenizer(data_path: str, save_dir: str, vocab_size: int = 8192,
                   batch_size: int = 10000, test_mode: bool = True, max_lines: int = None) -> None:
    """训练并保存自定义tokenizer - 使用批处理避免内存溢出"""
    
    print("=" * 60)
    print(f"开始训练 Tokenizer")
    print(f"数据路径: {data_path}")
    print(f"保存目录: {save_dir}")
    print(f"词汇表大小: {vocab_size}")
    print(f"批处理大小: {batch_size}")
    print(f"测试模式: {test_mode}")
    print(f"最大行数: {max_lines if max_lines else '全部'}")
    print("=" * 60)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 初始化tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = NFKC()  # 添加文本规范化
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    # 配置特殊token
    special_tokens = [
        "<unk>", 
        "<s>", 
        "</s>", 
        "<|im_start|>", 
        "<|im_end|>"
    ]

    # 配置训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        continuing_subword_prefix="##"  # 添加子词前缀
    )

    # 训练tokenizer
    print(f"开始训练 Tokenizer...")
    start_time = time.time()
    
    # 根据 test_mode 或显式 max_lines 控制读取规模
    effective_max_lines = 100000 if test_mode and max_lines is None else max_lines
    if effective_max_lines:
        print(f"限定最大行数: {effective_max_lines:,}")
    iterator = batch_iterator(data_path, batch_size=batch_size, max_lines=effective_max_lines)
    
    # 估算总行数用于进度显示（可选）
    # 这里我们不知道确切行数，所以不传递length参数
    tokenizer.train_from_iterator(iterator, trainer=trainer)
    
    training_time = time.time() - start_time
    print(f"Tokenizer 训练完成！耗时: {training_time:.1f}s")
    
    # 验证特殊token映射
    try:
        # 获取token映射
        vocab = tokenizer.get_vocab()
        print("\n特殊token映射验证:")
        for token in special_tokens:
            if token in vocab:
                print(f"  {token}: {vocab[token]}")
            else:
                print(f"  {token}: 未找到!")
    except Exception as e:
        print(f"特殊token验证错误: {e}")

    # 保存tokenizer文件
    tokenizer_file = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_file)
    
    # 创建配置文件
    create_tokenizer_config(save_dir)
    
    print("\n" + "=" * 60)
    print(f"✅ Tokenizer 训练完成!")
    print(f"保存位置: {save_dir}")
    print(f"词汇表大小: {len(tokenizer.get_vocab())}")
    print(f"总耗时: {training_time:.1f}s")
    print("=" * 60)

def quick_test(data_path: str, test_lines: int = 10000):
    """快速测试数据读取"""
    print("=" * 60)
    print("快速测试数据读取...")
    print("=" * 60)
    
    count = 0
    total_chars = 0
    start_time = time.time()
    
    with open(data_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            if i > test_lines:
                break
                
            try:
                data = json.loads(line)
                text = data.get('text', '')
                if text:
                    count += 1
                    total_chars += len(text)
                    
                if i % 1000 == 0:
                    elapsed = time.time() - start_time
                    print(f"已读取 {i:,} 行，有效 {count:,} 行，平均长度: {total_chars//max(count,1):,} 字符")
                    
            except Exception as e:
                continue
    
    elapsed = time.time() - start_time
    print("\n测试结果:")
    print(f"总行数: {test_lines:,}")
    print(f"有效行数: {count:,}")
    print(f"总字符数: {total_chars:,}")
    print(f"平均每行字符数: {total_chars//max(count,1):,}")
    print(f"读取速度: {test_lines/elapsed:.0f} 行/秒")
    print(f"耗时: {elapsed:.1f}s")

def eval_tokenizer(tokenizer_path: str) -> None:
    """评估tokenizer功能"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # 测试基本属性
    print("\n=== Tokenizer基本信息 ===")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Special tokens: {tokenizer.all_special_tokens}")
    
    # 测试简单编码解码
    print("\n=== 编码解码测试 ===")
    test_text = "Hello, world! 你好，世界！"
    encoded = tokenizer(test_text)
    decoded = tokenizer.decode(encoded["input_ids"])
    print(f"原始文本: {test_text}")
    print(f"Token IDs: {encoded['input_ids'][:10]}...")  # 只显示前10个
    print(f"解码文本: {decoded}")
    
    # 测试特殊token
    print("\n=== 特殊token测试 ===")
    print(f"<unk> token id: {tokenizer.convert_tokens_to_ids('<unk>')}")
    print(f"<|im_start|> token id: {tokenizer.convert_tokens_to_ids('<|im_start|>')}")

def main():
    # 配置路径
    data_path = "/root/data-fs/Data/seq_monkey_datawhale.jsonl"
    save_dir = "/root/Happy-llm/happy-llm-project/src/tokenization"
    
    print("1. 先进行快速数据测试")
    quick_test(data_path, test_lines=10000)
    
    print("\n2. 开始训练tokenizer（测试模式）")
    train_tokenizer(
        data_path=data_path,
        save_dir=save_dir,
        vocab_size=32768,
        batch_size=10000,
        test_mode=True  # 先使用测试模式
    )
    
    print("\n3. 评估tokenizer")
    eval_tokenizer(save_dir)

if __name__ == '__main__':
    main()