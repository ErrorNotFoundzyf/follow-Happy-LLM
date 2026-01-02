import torch 
import torch.nn as nn
import sys
from pathlib import Path
import math
import torch.nn.functional as F

# 获取当前文件的绝对路径
current_file = Path(__file__).resolve()
# 获取项目根目录：happy-llm-project/src/layers → happy-llm-project
project_root = current_file.parent.parent.parent

# 将项目根目录添加到Python路径
sys.path.insert(0, str(project_root))

# 现在可以导入
from src.llama2.config import ModelConfig

# 在 LLaMA2 模型中，我们需要将键和值的维度扩展到和查询的维度一样，这样才能进行注意力计算。
def repeat_kv(x:torch.Tensor,n_rep:int) ->torch.Tensor:
    # 获取输入张量的形状：批量大小、序列长度、键/值对头数的数量、每个头的维度大小
    bs,slen,n_kv_heads,head_dim = x.shape;

    #如果重复次数为1，则不需要重复，直接返回原始张量
    if n_rep==1: 
        return x;

    #对张量进行拓展和重塑操作以重复键值对，第三个维度是键值对头的维度
    return (
        x[:,:,:,None,:] #在第四个维度（头的维度前）添加一个新的维度
        .expand(bs,slen,n_kv_heads,n_rep,head_dim) #将新添加的维度扩展到n_rep大小，实现重复的效果
        .reshape(bs,slen,n_kv_heads*n_rep,head_dim) #重新塑形，合并键/值对头的数量和重复次数的维度
    )

#旋转嵌入：它能为注意力机制提供更强的上下文信息
# 注意：此处的dim应为 dim//n_head，因为我们是对每个head进行旋转嵌入
def precompute_freqs_cis(dim:int,end:int,theta:float=1000.0):
    # torch.arange(0,dim,2)[:(dim//2)].float()生成了一个从0开始，步长为2 的序列，长度为dim的一半
    # 然后每个元素除以dim，再提取theta的倒数，得到频率
    freqs = 1.0/(theta ** torch.arange(0,dim,2)[:(dim//2)].float() /dim)
    
    # 生成一个从0到end的时间序列，长度为end
    t = torch.arange(end,device=freqs.device)
    #计算外积，得到一个二维矩阵，每一行是t的元素乘以freqs的元素，得到二维矩阵freqs
    freqs = torch.outer(t,freqs).float()
    #计算频率的余弦值，得到实部
    freqs_cos = torch.cos(freqs)
    # 计算频率的正弦值，得到虚部
    freqs_sin = torch.sin(freqs)
    return freqs_cos,freqs_sin


# 构造调整张量形状的reshape_for_broadcast函数，
# 这个函数的主要目的是调整 freqs_cis 的形状，使其在进行广播操作时与 x 的维度对齐，从而能够进行正确的张量运算。
def reshape_for_broadcast(freqs_cis:torch.Tensor,x:torch.Tensor):
    # 获取x的维度数
    ndim = x.ndim

    #断言，确保1在x的维度范围内
    assert 0<=1 <ndim

    #断言，确保freqs_cis的形状与x的第二维和最后一维相同
    assert freqs_cis.shape == (x.shape[1],x.shape[-1])

    #构造一个新的形状，除了第二维和最后意味，其他维度都为1，这样做是为了能够将freqs_cis与x进行广播操作
    shape = [d if i==1 or i == ndim - 1 else 1 for i,d in enumerate(x.shape)]

    #将freqs_cis调整为新的形状，并返回
    return freqs_cis.view(shape)

#实现完整版的旋转嵌入
def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:

    # 将查询和键张量转换为浮点数，并重塑形状以分离实部和虚部
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # 重新塑形频率张量以进行广播
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # 应用旋转，分别计算旋转后的实部和虚部
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # 将最后两个维度合并，并还原为原始张量的形状
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)


# xq = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim
# xk = torch.randn(1, 50, 6, 48) # bs, seq_len, dim//n_head, n_head_dim

# # 使用 precompute_freqs_cis 函数获取 sin和cos
# cos, sin = precompute_freqs_cis(288//6, 50)
# print(cos.shape, sin.shape)
# xq_out, xk_out = apply_rotary_emb(xq, xk, cos, sin)

# xq_out.shape, xk_out.shape


#组装Attention结构
class Attention(nn.Module):
    def __init__ (self, args:ModelConfig):
        super().__init__()
        #根据是否指定n_kv_heads,确定用于键(key)和值(value)的头的数量。
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        #确保总头数可以被键值头数整除
        assert args.n_heads % self.n_kv_heads == 0

        #模型并行处理的大小，默认为1
        model_parallel_size = 1
        #本地计算头数，等于总头数除以模型并行处理大小
        self.n_local_heads = args.n_heads // model_parallel_size
        #本地键值头数，等于键值头数除以模型并行处理的大小
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        #重复次数，用于扩展键和值的尺寸
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        #每个头的维度，等于模型维度除以头的总数
        self.head_dim = args.dim //args.n_heads

        #定义权重矩阵
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        #输出权重矩阵
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim,bias=False)

        #定义dropout
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        #保存dropout概率
        self.dropout = args.dropout

        #检查是否实用Flash Attention 
        self.flash = hasattr(torch.nn.functional,'scaled_dot_product_attention')
        if not self.flash:
            #若不支持Flash Attention ，则实用手动实现的注意力机制，并设置mask.
            #创建一个上三角矩阵，用于遮蔽未来信息
            mask = torch.full((1,1,args.max_seq_len,args.max_seq_len),float("-inf"))
            mask = torch.triu(mask,diagonal=1)
            #注册为模型的缓冲区
            self.register_buffer("mask",mask)

    def forward(self, x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor):
            # 获取批次大小和序列长度，[batch_size, seq_len, dim]
            bsz, seqlen, _ = x.shape

            # 计算查询（Q）、键（K）、值（V）。
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            # 调整形状以适应头的维度。
            xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
            xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

            # 应用旋转位置嵌入（RoPE）。
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

            # 对键和值进行扩展以适应重复次数。
            xk = repeat_kv(xk, self.n_rep)
            xv = repeat_kv(xv, self.n_rep)

            # 将头作为批次维度处理。
            xq = xq.transpose(1, 2)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

            # 根据是否支持Flash Attention，选择实现方式。
            if self.flash:
                # 使用Flash Attention。
                output = torch.nn.functional.scaled_dot_product_attention(xq, xk, xv, attn_mask=None, dropout_p=self.dropout if self.training else 0.0, is_causal=True)
            else:
                # 使用手动实现的注意力机制。
                scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
                assert hasattr(self, 'mask')
                scores = scores + self.mask[:, :, :seqlen, :seqlen]
                scores = F.softmax(scores.float(), dim=-1).type_as(xq)
                scores = self.attn_dropout(scores)
                output = torch.matmul(scores, xv)

            # 恢复时间维度并合并头。
            output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

            # 最终投影回残差流。
            output = self.wo(output)
            output = self.resid_dropout(output)
            return output


# # 创建Attention实例
# args = ModelConfig(
#     dim=4096,           # 模型维度
#     n_heads=32,         # 注意力头的数量
#     n_kv_heads=None,    # 键值头数量（如果为None，则使用n_heads）
#     dropout=0.1,        # dropout概率
#     max_seq_len=2048    # 最大序列长度
# )

# attention_model = Attention(args)

# # 模拟输入数据
# batch_size = 1
# seq_len = 50  # 假设实际使用的序列长度为50
# dim = args.dim
# x = torch.rand(batch_size, seq_len, dim)  # 随机生成输入张量
# # freqs_cos = torch.rand(seq_len, dim // 2)  # 模拟cos频率，用于RoPE
# # freqs_sin = torch.rand(seq_len, dim // 2)  # 模拟sin频率，用于RoPE

# freqs_cos, freqs_sin = precompute_freqs_cis(dim//args.n_heads, seq_len)

# # 运行Attention模型
# output = attention_model(x, freqs_cos, freqs_sin)

# # attention出来之后的形状 依然是[batch_size, seq_len, dim]
# print("Output shape:", output.shape)
