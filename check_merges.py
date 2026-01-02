import json

tokenizer_path = "/root/Happy-llm/happy-llm-project/src/tokenization/tokenizer.json"

with open(tokenizer_path, 'r') as f:
    data = json.load(f)

merges = data['model']['merges']
print(f"Type of merges: {type(merges)}")
if len(merges) > 0:
    print(f"First merge item: {merges[0]}")
    print(f"Type of first merge item: {type(merges[0])}")
