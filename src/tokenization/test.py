import json
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, normalizers

# 读取原始 json
with open("tokenizer.json", "r") as f:
    data = json.load(f)

vocab = data["model"]["vocab"]
raw_merges = data["model"]["merges"]

# 转换 merges 格式
# 如果 raw_merges 是 [["a", "b"], ["c", "d"]]，转为 [("a", "b"), ("c", "d")]
merges = [tuple(pair) for pair in raw_merges]

# 用当前环境重建
tokenizer = Tokenizer(models.BPE(vocab=vocab, merges=merges, unk_token="<unk>"))
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
tokenizer.decoder = decoders.ByteLevel()

# 补全特殊 token（从 added_tokens 里读）
if "added_tokens" in data:
    special_tokens = [t["content"] for t in data["added_tokens"] if t.get("special")]
    tokenizer.add_special_tokens(special_tokens)

# 保存
tokenizer.save("tokenizer_fixed.json")
print("修复完成，已保存为 tokenizer_fixed.json")