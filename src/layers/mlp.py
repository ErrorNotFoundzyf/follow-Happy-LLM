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

class MLP(nn.Module):
    def __init__(self, dim:int, hidden_dim:int, multiple_of: int,dropout :float):
        super().__init__()
        #如果没有指定隐藏层的维度，我们将其设置为输入维度的4倍
        #然后将其减少到2/3，最后确保它是multiple_of的倍数

        if(hidden_dim is None):
            hidden_dim = 4 * dim
            hidden_dim = int(2 * hidden_dim /3)
            hidden_dim = multiple_of * ((hidden_dim +multiple_of -1 ) // multiple_of)

        #定义第一层线性变换，从输入维度到隐藏维度
        self.w1 = nn.Linear(dim,hidden_dim, bias = False)
        #定义第二层线性变换，从隐藏维度到输入维度
        self.w2 = nn.Linear(hidden_dim,dim, bias = False)
        #定义第三层线性变换，从输入维度到隐藏维度
        self.w3 = nn.Linear(dim,hidden_dim, bias = False)
        #定义dropout层，用于防止过拟合
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        #前向传播函数，这里用到的是SwiGLU FFN
        #首先，输入通过第一层线性变换和SILU激活函数
        #然后，结果乘以输入x通过第三层线性变换的结构
        #最后，通过第二层线性变换和dropout层
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

# args = ModelConfig(
#     dim=4096,           # 模型维度
#     n_heads=32,         # 注意力头的数量
#     n_kv_heads=None,    # 键值头数量（如果为None，则使用n_heads）
#     dropout=0.1,        # dropout概率
#     max_seq_len=2048    # 最大序列长度
# )
 
# # 创建MLP实例
# mlp = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)
# # 随机生成数据
# x = torch.randn(1, 50, args.dim)
# # 运行MLP模型
# output = mlp(x)
# print(output.shape)
