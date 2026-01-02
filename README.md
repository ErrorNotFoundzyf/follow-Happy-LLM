# Happy-LLM Project

本仓库复现了 Datawhale [《第五章 动手搭建大模型》](https://datawhalechina.github.io/happy-llm/#/./chapter5/%E7%AC%AC%E4%BA%94%E7%AB%A0%20%E5%8A%A8%E6%89%8B%E6%90%AD%E5%BB%BA%E5%A4%A7%E6%A8%A1%E5%9E%8B) 中的核心流程，聚焦于从零实现 LLaMA 风格模型、手写训练脚本，并在自有 GPU 环境上完成数据预处理、分词器训练与预训练/SFT 试验。由于算力受限，最终模型效果仍有待提升，但完整的工程脚手架与实验记录已整理于此。

> 💡 **这是一个非常好的入门 LLM 的项目！** Happy-LLM 系列教程通过动手实践，帮助初学者深入理解大模型的内部机制。完成本项目后，我计划继续 follow 同系列的 [Happy Agent](https://datawhalechina.github.io/hello-agents/#/) 项目，一步步构建完整的 AI Agent 系统。

## 仓库结构

```
happy-llm-project/
├── data/
│   ├── raw/                  # 原始下载数据（需手动放置）
│   └── prepared/             # 数据处理脚本与生成的 jsonl
├── models/
│   ├── checkpoints/          # 训练过程中的临时权重
│   ├── llama2_from_scratch/  # 从零训练的模型结果
│   └── pretrain/             # 预训练阶段产出的权重
├── src/
│   ├── dataset.py            # 预训练/SFT 自定义 Dataset
│   ├── utils.py              # 预留的通用工具
│   ├── layers/               # 手写的注意力、MLP、RMSNorm 层
│   ├── llama2/               # LLaMA 配置与解码器堆叠逻辑
│   ├── tokenization/         # 自定义分词器训练与评估脚本
│   └── trainer/              # 预训练与 SFT 训练/推理入口
├── run_pretrain.sh           # 预训练示例 Shell 脚本
├── run_finetune.sh           # SFT 示例 Shell 脚本
├── requirements.txt          # 运行环境依赖
├── train.ipynb               # Notebook 版演示
└── README.md
```

每个子模块都以 PyTorch 为核心编写，没有调用 Hugging Face Transformers 提供的 TransformerBlock 等高层 API，方便学习底层细节。

## 复现流程

### 1. 准备依赖与运行环境 ⚙️

- 建议使用 Linux + NVIDIA GPU；我们的实验环境为 H200（141GB 显存），Python 3.10。
- 推荐使用 Conda 创建隔离环境：

```bash
conda create -n happy-llm python=3.10 -y
conda activate happy-llm
pip install -r requirements.txt
```

### 2. 数据收集 📦

1. 预训练语料：Mobvoi「出去问问猴子」开放语料（mobvoi_seq_monkey_general_open_corpus.jsonl），共 28,998,098 行中英文混合对话/问答文本。
2. 指令微调语料：Belle Group 的对话数据（train_3.5M_CN.json），包含多轮 instruction-following 会话。

请将原始文件放入 data/raw/ 或 script 中指定的服务器路径，确保 data/prepared/data_prepare.py 可访问。

### 3. 数据预处理 🧹

使用 data/prepared/data_prepare.py 将原始语料整理为 jsonl：

```
python data/prepared/data_prepare.py
```

- 预训练数据会被按 512 字符切块并写到 seq_monkey_datawhale.jsonl，用于语言模型训练。
- SFT 数据被转换为标准 messages 列表并保存为 BelleGroup_sft.jsonl。

输出文件默认位于 /root/data-fs/Data/，可按需修改脚本中的输入/输出路径常量。

### 4. 分词器训练 🧩

src/tokenization/tokenizer.py 提供了基于 Hugging Face Tokenizers 的训练流程，可通过修改 main() 中的路径后运行：

```
python src/tokenization/tokenizer.py
```

在 H200 环境中，我们针对出去问问猴子语料的前 1,000,000 行训练过一个 32k 词表的 BPE 分词器，但效果欠佳。由于显存与内存限制，最终实验改用教程作者提供的 tokenizer（已放置在 src/tokenization/ 目录下）。

### 5. 预训练 🚀

以教程设置为基础，在 H200 上运行如下命令（训练约 14 小时，尚未产出理想效果）：

```bash
export TOKENIZERS_PARALLELISM=1
python src/trainer/pretrain.py \
  --data_path /root/data-fs/Data/seq_monkey_datawhale.jsonl \
  --out_dir /root/data-fs/LLM_model/My_llama_pretrain \
  --batch_size 256 \
  --accumulation_steps 2 \
  --dim 1024 --n_layers 18 \
  --vocab_size 6144 \
  --max_seq_len 256 \
  --epochs 1 \
  --num_workers 24 \
  --log_interval 200 \
  --save_interval 5000
```

训练完后，可用以下脚本快速抽样：

```bash
export TOKENIZERS_PARALLELISM=1
python src/trainer/infer_pretrain.py \
  --ckpt /root/data-fs/LLM_model/My_llama_pretrain/pretrain_1024_18_6144.pth \
  --tokenizer_dir src/tokenization \
  --prompt "请简要说明Transformer的核心思想" \
  --device cuda:0 \
  --dim 1024 --n_layers 18 --vocab_size 6144 --max_seq_len 512 \
  --max_new_tokens 128 --temperature 0.7 --top_k 50
```

> 现阶段模型生成能力有限，主要用于验证训练脚本完整性。

### 6. 指令微调（SFT） 🎯

当前仓库尚未提供可用的预训练权重，因此 fine_tune.py 会在随机初始化参数上直接微调：

```bash
export TOKENIZERS_PARALLELISM=1
python src/trainer/fine_tune.py \
  --data_path /root/data-fs/Data/BelleGroup_sft.jsonl \
  --out_dir /root/data-fs/LLM_model/My_llama_sft \
  --load_ckpt "" \
  --batch_size 24 \
  --accumulation_steps 2 \
  --dim 768 \
  --n_layers 12 \
  --vocab_size 6144 \
  --max_seq_len 512 \
  --epochs 1 \
  --num_workers 8 \
  --log_interval 100 \
  --save_interval 10000
```

推理脚本：

```bash
export TOKENIZERS_PARALLELISM=1
python src/trainer/infer_sft.py \
  --ckpt /root/data-fs/LLM_model/My_llama_sft/sft_768_12_6144_final.pth \
  --tokenizer_dir src/tokenization \
  --prompt "帮我写一首关于春天的短诗" \
  --device cuda:0 \
  --dim 768 --n_layers 12 --vocab_size 6144 --max_seq_len 512 \
  --max_new_tokens 128 --temperature 0.7 --top_k 50
```

由于缺乏稳定的预训练基座，加之训练时长有限，SFT 结果尚无法流畅回答问题。

## 源码概览 🛠️

- src/dataset.py：实现流式 jsonl 读取、offset 索引缓存的 PretrainDataset，以及兼容多种指令格式的 SFTDataset。
- src/tokenization/：包含批量读取大语料训练 BPE 的脚本、tokenizer 评估工具和作者提供的最终 tokenizer 配置。
- src/layers/attention.py：手写多头注意力、RoPE 计算与 Flash Attention 兼容逻辑。
- src/layers/mlp.py：实现 SwiGLU 前馈网络并支持 multiple_of 对齐。
- src/layers/rmsnorm.py：实现 RMSNorm。
- src/llama2/config.py：基于 PretrainedConfig 管理模型超参。
- src/llama2/decoder.py：堆叠自定义解码层，实现残差 + RMSNorm 架构。
- src/llama2/model.py：封装 Transformer 主体（包含生成接口）。
- src/trainer/pretrain.py：纯 PyTorch 的语言模型训练循环，支持梯度累积、AMP、断点保存。
- src/trainer/fine_tune.py：SFT 训练管线，包含自适应学习率调度、loss mask 计算。
- src/trainer/infer_pretrain.py / infer_sft.py：简易推理脚本，便于快速验证模型。

以上模块均为手写实现，旨在理解 Transformer 细节，仅在 tokenizer 相关任务中依赖 Hugging Face Tokenizers/Transformers 提供的基础设施。

## 实验记录与限制 📈

- 分词器：H200 上自训 1M 行样本效果欠佳，最终采用作者提供的 tokenizer。
- 训练环境：H200 服务器，Python 3.10，依赖列表见 requirements.txt。
- 结果：预训练持续约 14 小时仍未收敛到可用程度，SFT 阶段同样无法生成合理回答。主要瓶颈是训练时长与显存开销，后续可考虑缩小模型维度或增加训练轮数。

欢迎在此基础上继续优化模型结构、训练策略或引入增量数据，提升生成质量。

2025 年 12 月 29 日动工，12 月 31 日收官。感谢dx老师给我提供这么好的算力。这是第一次手搓大模型，虽然效果一般，但过程非常开心，全程都以很痴迷的状态去学习，这是一次难得的尝试与入门，希望自己始终保持好奇心，呼吸着欣喜的空气，做自己感兴趣的事情。