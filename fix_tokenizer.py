import json
from tokenizers import Tokenizer

tokenizer_path = "/root/Happy-llm/happy-llm-project/src/tokenization/tokenizer.json"
fixed_tokenizer_path = "/root/Happy-llm/happy-llm-project/src/tokenization/tokenizer_fixed.json"

try:
    with open(tokenizer_path, 'r') as f:
        data = json.load(f)
    
    print("JSON loaded successfully.")
    
    if 'model' in data:
        model = data['model']
        print(f"Model keys: {list(model.keys())}")
        
        # 尝试移除可能导致兼容性问题的字段
        keys_to_remove = ['fuse_unk', 'byte_fallback', 'ignore_merges']
        for key in keys_to_remove:
            if key in model:
                print(f"Removing key: {key}")
                del model[key]
        
        # 保存修复后的文件
        with open(fixed_tokenizer_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Saved fixed tokenizer to {fixed_tokenizer_path}")
        
        # 尝试加载
        try:
            tokenizer = Tokenizer.from_file(fixed_tokenizer_path)
            print("Successfully loaded fixed tokenizer!")
            print(f"Vocab size: {tokenizer.get_vocab_size()}")
        except Exception as e:
            print(f"Failed to load fixed tokenizer: {e}")
            
except Exception as e:
    print(f"Error processing json: {e}")
