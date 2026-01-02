import json
from tokenizers import Tokenizer

tokenizer_path = "/root/Happy-llm/happy-llm-project/src/tokenization/tokenizer.json"
fixed_tokenizer_path = "/root/Happy-llm/happy-llm-project/src/tokenization/tokenizer.json" # 直接覆盖，或者先备份

# 备份
import shutil
shutil.copy(tokenizer_path, tokenizer_path + ".bak")

try:
    with open(tokenizer_path, 'r') as f:
        data = json.load(f)
    
    print("JSON loaded successfully.")
    
    if 'model' in data:
        model = data['model']
        
        # 1. 修复 merges 格式
        if 'merges' in model:
            merges = model['merges']
            if len(merges) > 0 and isinstance(merges[0], list):
                print("Converting merges from list of lists to list of strings...")
                new_merges = [f"{pair[0]} {pair[1]}" for pair in merges]
                model['merges'] = new_merges
                print("Merges converted.")
        
        # 2. 移除可能导致兼容性问题的字段
        keys_to_remove = ['fuse_unk', 'byte_fallback', 'ignore_merges']
        for key in keys_to_remove:
            if key in model:
                print(f"Removing key: {key}")
                del model[key]
        
        # 保存修复后的文件
        with open(fixed_tokenizer_path, 'w') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"Saved fixed tokenizer to {fixed_tokenizer_path}")
        
        # 尝试加载
        try:
            tokenizer = Tokenizer.from_file(fixed_tokenizer_path)
            print("Successfully loaded fixed tokenizer!")
            print(f"Vocab size: {tokenizer.get_vocab_size()}")
        except Exception as e:
            print(f"Failed to load fixed tokenizer: {e}")
            # 如果失败，恢复备份
            shutil.copy(tokenizer_path + ".bak", tokenizer_path)
            print("Restored original file due to load failure.")
            
except Exception as e:
    print(f"Error processing json: {e}")
