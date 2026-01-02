import json
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class PretrainDataset(Dataset):
    """
    内存友好的预训练数据集：
    - 使用行偏移索引（.idx.npy）随机访问每一行，避免将整个数据文件读入内存。
    - 支持 DataLoader 的 shuffle（通过索引随机采样）。
    - 单次样本读取仅打开文件并读取一行，极大降低系统内存占用。
    """

    def __init__(self, data_path, tokenizer, max_length=512, index_path: str = None):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0

        # 索引文件路径
        self.index_path = index_path or (data_path + ".idx.npy")

        # 若存在已构建的偏移索引，则内存映射加载；否则先构建再加载
        if os.path.exists(self.index_path):
            self.offsets = np.load(self.index_path, mmap_mode='r')
        else:
            offsets = []
            offset = 0
            with open(self.data_path, 'rb') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    offsets.append(offset)
                    offset += len(line)
            # 保存为 .npy，之后以内存映射方式读取，避免占用常驻内存
            np.save(self.index_path, np.asarray(offsets, dtype=np.int64))
            self.offsets = np.load(self.index_path, mmap_mode='r')

    def __len__(self):
        return int(self.offsets.shape[0])

    def __getitem__(self, index: int):
        # 读取指定偏移处的一行
        with open(self.data_path, 'rb') as f:
            f.seek(int(self.offsets[index]))
            line = f.readline().decode('utf-8').rstrip('\n')

        sample = json.loads(line)
        text = f"{self.tokenizer.bos_token}{sample['text']}"
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        # 尽量使用 tokenizer 的 pad_token_id；没有则退回 eos/0
        self.padding = getattr(tokenizer, "pad_token_id", None)
        if self.padding is None:
            self.padding = getattr(tokenizer, "eos_token_id", 0) or 0
        with open(data_path, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __len__(self):
        return len(self.data)

    def _render_chat(self, sample):
        """兼容无 apply_chat_template 的 tokenizer，手工构建对话模板。

        优先使用 tokenizer 的 apply_chat_template；若无则退回简单模板，
        支持 Belle 风格 {instruction,input,output} 或 {messages/ conversation}。
        """
        if hasattr(self.tokenizer, "apply_chat_template"):
            try:
                return self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
            except Exception:
                pass

        # 手工模板
        start = "<|im_start|>"
        end = "<|im_end|>"

        # Belle: instruction + optional input + output
        if isinstance(sample, dict) and "instruction" in sample:
            user = sample.get("instruction", "")
            if sample.get("input"):
                user = f"{user}\n{sample['input']}"
            assistant = sample.get("output", sample.get("response", ""))
            return f"{start}user\n{user}{end}\n{start}assistant\n{assistant}{end}\n"

        # 通用 messages/conversation: list of {role, content}
        conv = sample.get("messages") if isinstance(sample, dict) else None
        if conv is None and isinstance(sample, dict):
            conv = sample.get("conversation")
        if conv and isinstance(conv, list):
            chunks = []
            for m in conv:
                role = m.get("role", "user")
                content = m.get("content", "")
                chunks.append(f"{start}{role}\n{content}{end}\n")
            return "".join(chunks)

        # 回退：直接用字符串
        if isinstance(sample, str):
            return f"{start}user\n{sample}{end}\n"
        return f"{start}user\n{json.dumps(sample, ensure_ascii=False)}{end}\n"

    def generate_loss_mask(self, input_ids):
        # 生成 loss mask, 0 表示不计算损失, 1 表示计算损失
        mask = [0] * len(input_ids)
        # <|im_start|>assistant\n
        a_sequence = [3, 1074, 537, 500, 203]  # <|im_start|>assistant\n
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0
        
        while i <= n - a_length:
            # 检查当前位置是否匹配目标子序列
            match = True
            for k in range(a_length):
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break
            if match:
                # 从子序列结束的位置开始查找第一个4, 4 为 <|im_end|> EOS id
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == 4:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j  # 结束位置设为j（包含4）
                    # 标记区间为1（包括start到end）
                    if start <= end:
                        for pos in range(start, end + 1):
                            if pos < len(mask):
                                mask[pos] = 1
                # 跳过当前子序列，避免重叠匹配
                i += a_length
            else:
                i += 1
        return mask

    def __getitem__(self, index: int):
        sample = json.loads(self.data[index])
        text = self._render_chat(sample)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 简化：对非 padding 位置计算损失，避免因模板 token id 不匹配导致全 0 掩码
        loss_mask = [0 if tid == self.padding else 1 for tid in input_id]

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)
