#!/usr/bin/env python3
# eval_tokenizer.py
"""
评估自定义tokenizer的功能和质量
使用方法: python eval_tokenizer.py [tokenizer_path]
"""

import sys
import os
from pathlib import Path
import json

def load_tokenizer(tokenizer_path: str):
    """加载tokenizer"""
    from transformers import AutoTokenizer
    
    print(f"正在加载tokenizer: {tokenizer_path}")
    
    # 检查路径是否存在
    if not os.path.exists(tokenizer_path):
        print(f"错误: 路径不存在 {tokenizer_path}")
        # 尝试寻找可能的路径
        possible_paths = [
            tokenizer_path,
            os.path.join(tokenizer_path, "tokenizer.json"),
            os.path.join(tokenizer_path, "tokenizer_config.json"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                print(f"找到文件: {path}")
                tokenizer_path = os.path.dirname(path) if os.path.isfile(path) else path
                break
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print("✅ Tokenizer加载成功!")
        return tokenizer
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        
        # 尝试直接使用tokenizers库
        try:
            from tokenizers import Tokenizer
            tokenizer_hf = Tokenizer.from_file(os.path.join(tokenizer_path, "tokenizer.json"))
            print("⚠️  使用tokenizers库加载成功，但功能有限")
            return tokenizer_hf
        except:
            return None

def test_basic_properties(tokenizer):
    """测试基本属性"""
    print("\n" + "="*60)
    print("1. 基本属性测试")
    print("="*60)
    
    try:
        print(f"词汇表大小: {len(tokenizer)}")
    except:
        print("无法获取词汇表大小")
    
    # 特殊token
    if hasattr(tokenizer, 'all_special_tokens'):
        print(f"特殊token: {tokenizer.all_special_tokens}")
    if hasattr(tokenizer, 'all_special_ids'):
        print(f"特殊token IDs: {tokenizer.all_special_ids}")
    
    # 检查关键特殊token
    special_tokens_to_check = ['<unk>', '<s>', '</s>', '<|im_start|>', '<|im_end|>', '[PAD]', '[CLS]', '[SEP]', '[MASK]']
    
    print("\n特殊token映射:")
    for token in special_tokens_to_check:
        try:
            token_id = tokenizer.convert_tokens_to_ids(token)
            if token_id != tokenizer.unk_token_id:
                print(f"  {token}: {token_id}")
        except:
            pass

def test_encoding_decoding(tokenizer):
    """测试编码解码"""
    print("\n" + "="*60)
    print("2. 编码解码测试")
    print("="*60)
    
    test_cases = [
        "Hello, world!",
        "你好，世界！",
        "こんにちは、世界！",
        "안녕하세요, 세계!",
        "1234567890",
        "Hello 你好 123",
        "这是一段较长的中文文本，用于测试tokenizer的分词效果。",
        "import numpy as np\nx = np.array([1, 2, 3])",
        "The quick brown fox jumps over the lazy dog.",
        "机器学习、深度学习、自然语言处理是人工智能的重要方向。",
    ]
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {text[:50]}{'...' if len(text) > 50 else ''}")
        try:
            # 编码
            encoded = tokenizer.encode(text)
            if isinstance(encoded, dict):
                encoded_ids = encoded.get('input_ids', [])
            else:
                encoded_ids = encoded
            
            # 解码
            decoded = tokenizer.decode(encoded_ids, skip_special_tokens=False)
            
            print(f"  Token数量: {len(encoded_ids)}")
            print(f"  Token IDs (前10): {encoded_ids[:10]}...")
            
            # 检查是否保留原意
            decoded_clean = tokenizer.decode(encoded_ids, skip_special_tokens=True)
            if text.strip() == decoded_clean.strip():
                print(f"  ✅ 解码一致")
            else:
                print(f"  ⚠️  解码差异")
                print(f"     原始: {text}")
                print(f"     解码: {decoded_clean}")
                
        except Exception as e:
            print(f"  ❌ 错误: {e}")

def test_chat_template(tokenizer):
    """测试聊天模板"""
    print("\n" + "="*60)
    print("3. 聊天模板测试")
    print("="*60)
    
    if not hasattr(tokenizer, 'apply_chat_template'):
        print("该tokenizer不支持聊天模板")
        return
    
    messages = [
        {"role": "system", "content": "你是一个AI助手。"},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "I'm fine, thank you. and you?"},
        {"role": "user", "content": "I'm good too."},
        {"role": "assistant", "content": "That's great to hear!"},
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        print("生成的prompt:")
        print("-" * 40)
        print(prompt)
        print("-" * 40)
        
        # 编码测试
        encoded = tokenizer(prompt, truncation=True, max_length=256)
        decoded = tokenizer.decode(encoded["input_ids"], skip_special_tokens=False)
        
        print(f"\n编码后token数量: {len(encoded['input_ids'])}")
        print(f"特殊token保留: {'✅' if '<|im_start|>' in decoded and '<|im_end|>' in decoded else '❌'}")
        
    except Exception as e:
        print(f"聊天模板测试失败: {e}")

def test_special_tokens(tokenizer):
    """测试特殊token处理"""
    print("\n" + "="*60)
    print("4. 特殊token处理测试")
    print("="*60)
    
    special_test_cases = [
        ("<|im_start|>user\nHello<|im_end|>", "基本特殊token"),
        ("<|im_start|>system\n你是一个助手<|im_end|>\n<|im_start|>user\n你好<|im_end|>", "多轮对话"),
        ("<unk>这个词应该被特殊处理", "未知词处理"),
        ("<s>开始文本</s>结束", "开始结束标记"),
        ("这是一段<|im_start|>user\n包含特殊token<|im_end|>的文本", "混合文本"),
    ]
    
    for test_text, description in special_test_cases:
        print(f"\n测试: {description}")
        print(f"  文本: {test_text}")
        
        try:
            encoded = tokenizer.encode(test_text)
            if isinstance(encoded, dict):
                encoded_ids = encoded.get('input_ids', [])
            else:
                encoded_ids = encoded
            
            decoded = tokenizer.decode(encoded_ids, skip_special_tokens=False)
            
            # 检查特殊token是否被正确识别
            special_tokens_present = []
            for token in ['<|im_start|>', '<|im_end|>', '<unk>', '<s>', '</s>']:
                if token in test_text and token in decoded:
                    special_tokens_present.append(token)
            
            if special_tokens_present:
                print(f"  ✅ 特殊token保留: {special_tokens_present}")
            else:
                print(f"  ⚠️  特殊token可能被转换")
                
            print(f"  Token数量: {len(encoded_ids)}")
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")

def test_vocabulary_coverage(tokenizer):
    """测试词汇表覆盖度"""
    print("\n" + "="*60)
    print("5. 词汇表覆盖度测试")
    print("="*60)
    
    test_words = [
        # 中文常见词
        "人工智能", "机器学习", "深度学习", "神经网络", "自然语言",
        "计算机", "编程", "算法", "数据", "模型",
        # 英文常见词
        "artificial", "intelligence", "machine", "learning", "deep",
        "neural", "network", "natural", "language", "processing",
        # 混合词
        "Python", "Java", "C++", "JavaScript", "HTML",
        # 数字和符号
        "123", "3.14", "2023-12-29", "100%", "test@example.com",
    ]
    
    print("常见词汇分词测试:")
    total_chars = 0
    total_tokens = 0
    
    for word in test_words:
        try:
            encoded = tokenizer.encode(word)
            if isinstance(encoded, dict):
                encoded_ids = encoded.get('input_ids', [])
            else:
                encoded_ids = encoded
            
            token_count = len(encoded_ids)
            chars = len(word)
            
            total_chars += chars
            total_tokens += token_count
            
            efficiency = chars / max(token_count, 1)
            
            if token_count == 1:
                print(f"  ✅ {word}: 1个token (效率: {efficiency:.1f} 字符/token)")
            elif token_count <= 3:
                print(f"  ⚠️  {word}: {token_count}个token (效率: {efficiency:.1f} 字符/token)")
            else:
                print(f"  ❌ {word}: {token_count}个token (效率: {efficiency:.1f} 字符/token)")
                
        except:
            print(f"  ❌ {word}: 编码失败")
    
    if total_tokens > 0:
        avg_efficiency = total_chars / total_tokens
        print(f"\n平均效率: {avg_efficiency:.2f} 字符/token")
        print(f"总字符数: {total_chars}, 总token数: {total_tokens}")

def test_oov_handling(tokenizer):
    """测试OOV（未登录词）处理"""
    print("\n" + "="*60)
    print("6. 未登录词处理测试")
    print("="*60)
    
    # 故意构造一些可能不在词汇表中的词
    oov_words = [
        "区块链技术应用",  # 可能被拆分的复合词
        "transformer模型",  # 混合词
        "COVID-19疫情",  # 特殊术语
        "中华人民共和国",  # 长专有名词
        "supercalifragilisticexpialidocious",  # 长英文单词
        "魑魅魍魉饕餮",  # 生僻中文
        "123!@#$%^&*()",  # 特殊符号
    ]
    
    for word in oov_words:
        try:
            encoded = tokenizer.encode(word)
            if isinstance(encoded, dict):
                encoded_ids = encoded.get('input_ids', [])
            else:
                encoded_ids = encoded
            
            # 检查是否包含<unk>
            unk_id = getattr(tokenizer, 'unk_token_id', None)
            if unk_id and unk_id in encoded_ids:
                print(f"  ⚠️  {word}: 包含<unk> token")
            else:
                print(f"  ✅ {word}: {len(encoded_ids)}个token")
                
        except Exception as e:
            print(f"  ❌ {word}: 错误 - {e}")

def save_test_report(tokenizer, tokenizer_path, output_file="tokenizer_test_report.txt"):
    """保存测试报告"""
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        print("Tokenizer测试报告")
        print("="*60)
        print(f"测试时间: {sys.argv[0]}")
        print(f"Tokenizer路径: {tokenizer_path}")
        
        # 重新运行测试但输出到StringIO
        test_basic_properties(tokenizer)
        test_encoding_decoding(tokenizer)
        test_chat_template(tokenizer)
        test_special_tokens(tokenizer)
        test_vocabulary_coverage(tokenizer)
        test_oov_handling(tokenizer)
    
    report = f.getvalue()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ 测试报告已保存到: {output_file}")

def main():
    """主函数"""
    # 确定tokenizer路径
    if len(sys.argv) > 1:
        tokenizer_path = sys.argv[1]
    else:
        # 默认使用当前目录
        tokenizer_path = os.path.dirname(os.path.abspath(__file__))
        print(f"未指定路径，使用当前目录: {tokenizer_path}")
    
    print("="*60)
    print("Tokenizer评估工具")
    print("="*60)
    
    # 加载tokenizer
    tokenizer = load_tokenizer(tokenizer_path)
    if tokenizer is None:
        print("无法加载tokenizer，请检查路径")
        return
    
    # 运行所有测试
    test_basic_properties(tokenizer)
    test_encoding_decoding(tokenizer)
    test_chat_template(tokenizer)
    test_special_tokens(tokenizer)
    test_vocabulary_coverage(tokenizer)
    test_oov_handling(tokenizer)
    
    # 保存测试报告
    save_test_report(tokenizer, tokenizer_path)
    
    print("\n" + "="*60)
    print("✅ 评估完成!")
    print("="*60)

if __name__ == "__main__":
    main()