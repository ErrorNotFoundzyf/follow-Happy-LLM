# LLaMA2模型就是将DecoderLayer模块堆叠起来，构成一个完整的Transformer模型
import sys
from pathlib import Path
import math
import torch.nn.functional as F
import torch.nn as nn
import torch


# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录：happy-llm-project/src/layers → happy-llm-project
project_root = current_file.parent.parent.parent

# 将项目根目录添加到Python路径
sys.path.insert(0, str(project_root))

# 现在可以导入
from src.llama2.config import ModelConfig
from src.layers.mlp import MLP
from src.layers.attention import Attention
from src.layers.rmsnorm import RMSNorm
from src.layers.attention import precompute_freqs_cis
from src.llama2.decoder import DecoderLayer

from typing import Optional
from transformers import PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

class Transformer(PreTrainedModel):
    config_class = ModelConfig #配置类
    last_loss:Optional[torch.Tensor] #记录最后一次计算的损失

    def __init__(self, args:ModelConfig = None):
        super().__init__(args)
        #初始化模型参数
        self.args = args
        #词汇表大小
        self.vocab_size = args.vocab_size
        #层数
        self.n_layers = args.n_layers

        #词嵌入层
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        #Dropout层
        self.dropout = nn.Dropout(args.dropout)
        #Decoder层
        self.layers = torch.nn.ModuleList() #创建了一个可注册的、包含多个神经网络模块的列表对象
        for layer_id in range(args.n_layers):
            self.layers.append(DecoderLayer(layer_id, args))
        #归一化层
        self.norm = RMSNorm(args.dim,eps=args.norm_eps)
        #输出层
        self.output = nn.Linear(args.dim, args.vocab_size,bias=False)

        #将词嵌入层的权重与输出层的权重共享
        self.tok_embeddings.weight = self.output.weight

        #预计算相对位置嵌入的频率
        freqs_cos, freqs_sin = precompute_freqs_cis(self.args.dim // self.args.n_heads, self.args.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        #初始化所有权重
        self.apply(self._init_weights)
        #对残差投影进行特殊的缩放初始化
        for pn,p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p,mean=0.0,std=0.02/math.sqrt(2 * args.n_layers))

        #初始化最后一次前向传播的损失属性
        self.last_loss = None
        self.OUT = CausalLMOutputWithPast() #输出容器
        self.__no_split_modules = [name for name, _ in self.named_modules()] #部分个的模块列表


    def _init_weights(self, module):
        #初始化权重的函数
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,tokens:torch.Tensor, targets:Optional[torch.Tensor]=None,**kwargs) ->torch.Tensor:
        """
        - tokens: Optional[torch.Tensor], 输入 token 张量。
        - targets: Optional[torch.Tensor], 目标 token 张量。
        - kv_cache: bool, 是否使用键值缓存。
        - kwargs: 其他关键字参数。

        - self.OUT: CausalLMOutputWithPast, 包含 logits 和损失。
        
        输入:
            tokens: 已经token化的文本，如 [["今天", "天气", "很好"]]
            targets: 训练时的目标标签（推理时为None）

        输出:
            训练时: {"loss": tensor, "logits": tensor}
            推理时: {"logits": tensor}  # 预测的下一个token概率
        """

        if 'input_ids' in kwargs:
            tokens = kwargs['input_ids']
        if 'attention_mask' in kwargs:
            targets = kwargs['attention_mask']
        
        #前向传播函数
        _bsz, seqlen = tokens.shape
        #通过词嵌入层和Dropout层
        h = self.tok_embeddings(tokens)
        h = self.dropout(h)
        #获取相对位置嵌入的频率
        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        #通过Decoder层
        for layer in self.layers:
            h = layer(h,freqs_cos,freqs_sin)
        #通过归一化层
        h = self.norm(h)

        if targets is not None:
            #如果给定了目标，计算损失
            logits = self.output(h)
            self.last_loss = F.cross_entropy(logits.view(-1,logits.size(-1)), targets.view(-1),ignore_index=0,reduction='none')
        else:
            #推理时的小优化：只对最后一个位置的输出进行前向传播
            logits = self.output(h[:,[-1],:])
            self.last_loss = None
        
        # 设置输出
        self.OUT.__setitem__('logits',logits)
        self.OUT.__setitem__('last_loss',self.last_loss)
        return self.OUT

    @torch.inference_mode()
    def generate(self, idx, stop_id=None, max_new_tokens=256, temperature = 1.0,top_k =None):
        """
        给定输入序列 idx（形状为 (bz,seq_len) 的长整型张量），通过多次生成新 token 来完成序列。
        在 model.eval() 模式下运行。效率较低的采样版本，没有使用键k/v cache。
       
        输入:
            idx: 初始prompt的token序列，如 [[1, 45, 23, 78]]
            max_new_tokens: 要生成的新token数量
        
        输出:
            生成的完整token序列，包括输入的prompt
            如: [[1, 45, 23, 78, 99, 102, 56, ...]]
        """
        index = idx.shape[1]
        for _ in range(max_new_tokens):
             
            #如果序列上下文过长，截断它到最大长度
            idx_cond = idx if idx.size(1) <= self.args.max_seq_len else idx[:,-self.args.max_seq_len:]

            #前向传播获取序列中的最后一个位置的Logits
            logits = self(idx_cond).logits # 这行代码实际上隐式调用了 self.forward(idx_cond)
            logits = logits[:, -1, :] #只保留最后一个时间步的输出

            if temperature == 0.0:
                #选择最有可能的索引,贪婪解码
                _,idx_next = torch.topk(logits,k=1,dim=-1)
            else :
                #缩放 logits 并应用softmax
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k,logits.size(-1)))
                    logits[logits < v[:,[-1]]] = -float('Inf')
                probs = F.softmax(logits, dim = -1)
                idx_next = torch.multinomial(probs, num_samples=1)

            if idx_next == stop_id:
                break;

            #将采用的索引添加搭配序列中并继续
            idx = torch.cat((idx,idx_next),dim=1)

        return idx[:, index:] #只返回生成的token



              
# args = ModelConfig(
#     dim=768,           # 模型维度
#     n_heads=32,         # 注意力头的数量
#     n_kv_heads=None,    # 键值头数量（如果为None，则使用n_heads）
#     dropout=0.1,        # dropout概率
#     max_seq_len=2048    # 最大序列长度
# )       
            
# # LLaMA2Model.forward 接受两个参数，tokens和targets，其中tokens是输入的张量, 应为int类型
# x = torch.randint(0, 6144, (1, 50)) # [bs, seq_len]
# # 实例化LLaMA2Model
# model = Transformer(args=args)
# # 计算model的全部参数
# num_params = sum(p.numel() for p in model.parameters())
# print('Number of parameters:', num_params)

# out = model(x)
# print(out.logits.shape) # [batch_size, 1, vocab_size]
