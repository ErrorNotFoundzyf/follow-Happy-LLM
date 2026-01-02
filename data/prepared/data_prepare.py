import os
import json
from tqdm import tqdm

# 设置数据存储目录
DATA_DIR = "/root/data-fs/Data"
os.makedirs(DATA_DIR, exist_ok=True)

def process_pretrain_data():
    """处理预训练数据"""
    
    print("\n" + "=" * 50)
    print("开始处理预训练数据...")
    print("=" * 50)
    
    # 直接指定输入文件路径
    input_file = "/root/data-fs/Data/mobvoi_seq_monkey_general_open_corpus.jsonl"
    print(f"输入文件: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"错误: 未找到文件 {input_file}")
        return False
    
    def split_text(text, chunk_size=512):
        """将文本按指定长度切分成块"""
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    output_file = os.path.join(DATA_DIR, "seq_monkey_datawhale.jsonl")
    
    # 统计总行数
    print("正在统计行数...")
    total_lines = 0
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for _ in f:
                total_lines += 1
    except Exception as e:
        print(f"统计行数时出错: {e}")
        # 如果统计失败，给一个估计值或不显示进度条总数
        total_lines = None
    
    print(f"开始处理 {total_lines if total_lines else '未知'} 行数据...")
    
    count = 0
    with open(output_file, 'w', encoding='utf-8') as pretrain:
        with open(input_file, 'r', encoding='utf-8') as f:
            iterator = tqdm(f, total=total_lines, desc="处理预训练数据") if total_lines else f
            for line in iterator:
                try:
                    line = line.strip()
                    if not line:
                        continue
                    line_data = json.loads(line)
                    text = line_data.get('text', '')
                    if text:
                        chunks = split_text(text)
                        for chunk in chunks:
                            if chunk.strip():  # 跳过空块
                                pretrain.write(json.dumps({'text': chunk}, ensure_ascii=False) + '\n')
                                count += 1
                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    print(f"处理行时出错: {e}")
                    continue
    
    print(f"预训练数据处理完成，输出到: {output_file}")
    print(f"生成 {count} 条训练样本")
    
    return True

def process_sft_data():
    """处理SFT数据"""
    
    print("\n" + "=" * 50)
    print("开始处理SFT数据...")
    print("=" * 50)
    
    # 直接指定输入文件路径
    input_file = "/root/data-fs/Data/train_3.5M_CN.json"
    print(f"输入文件: {input_file}")
    
    if not os.path.exists(input_file):
        print(f"错误: 未找到文件 {input_file}")
        return False
    
    def convert_message(data):
        """将原始数据转换为标准格式"""
        message = [
            {"role": "system", "content": "你是一个AI助手"},
        ]
        for item in data:
            if item.get('from') == 'human':
                message.append({'role': 'user', 'content': item.get('value', '')})
            elif item.get('from') == 'gpt' or item.get('from') == 'assistant':
                message.append({'role': 'assistant', 'content': item.get('value', '')})
        return message
    
    output_file = os.path.join(DATA_DIR, "BelleGroup_sft.jsonl")
    
    # 检查文件格式并统计行数
    print("正在检查文件格式并统计行数...")
    is_json_array = False
    total_lines = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            # 读取第一个非空字符
            while True:
                char = f.read(1)
                if not char:
                    break
                if char.strip():
                    if char == '[':
                        is_json_array = True
                    break
            
            if is_json_array:
                print("检测到文件为 JSON 数组格式，将一次性加载（请确保内存充足）...")
            else:
                print("检测到文件为 JSONL 格式（每行一个JSON）...")
                f.seek(0)
                for _ in f:
                    total_lines += 1
    except Exception as e:
        print(f"检查文件时出错: {e}")
        return False

    count = 0
    with open(output_file, 'w', encoding='utf-8') as sft:
        if is_json_array:
            # 如果是 JSON 数组，一次性加载
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"加载了 {len(data)} 条数据，开始转换...")
                for item in tqdm(data, desc="处理SFT数据"):
                    try:
                        conversations = item.get('conversations', [])
                        if conversations:
                            message = convert_message(conversations)
                            if len(message) > 1:
                                sft.write(json.dumps(message, ensure_ascii=False) + '\n')
                                count += 1
                    except Exception as e:
                        continue
            except Exception as e:
                print(f"加载 JSON 数组失败: {e}")
                return False
        else:
            # 如果是 JSONL，逐行处理
            print(f"开始处理 {total_lines} 行数据...")
            with open(input_file, 'r', encoding='utf-8') as f:
                for line in tqdm(f, total=total_lines, desc="处理SFT数据"):
                    try:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        conversations = item.get('conversations', [])
                        if conversations:
                            message = convert_message(conversations)
                            if len(message) > 1:
                                sft.write(json.dumps(message, ensure_ascii=False) + '\n')
                                count += 1
                    except json.JSONDecodeError:
                        continue
                    except Exception as e:
                        continue
    
    print(f"SFT数据处理完成，输出到: {output_file}")
    print(f"生成 {count} 条SFT样本")
    
    return True

def main():
    """主函数"""
    
    print("开始数据处理流程...")
    print(f"数据将保存到: {DATA_DIR}")
    
    # 步骤1: 处理预训练数据
    if not process_pretrain_data():
        print("预训练数据处理失败")
    
    # 步骤2: 处理SFT数据
    if not process_sft_data():
        print("SFT数据处理失败")
    
    print("\n" + "=" * 50)
    print("🎉 所有数据处理完成!")
    print("=" * 50)
    print(f"生成的文件位置:")
    print(f"1. 预训练数据: {DATA_DIR}/seq_monkey_datawhale.jsonl")
    print(f"2. SFT数据: {DATA_DIR}/BelleGroup_sft.jsonl")
    print("\n下一步: 可以使用这些文件训练tokenizer了!")
    print("=" * 50)

if __name__ == "__main__":
    main()
