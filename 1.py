import torch
import numpy as np


class Net(torch.nn.Module):
    def forward(self, x):
        # 第一层神经网络
        z1 = 2 * x[0] + x[1]
        z2 = x[0] * 3 * x[2]
        z3 = -x[2]
        # 第二层神经网络
        u1 = u1 = torch.sin(z1)
        u2 = 2 * x[2] + z2
        u3 = 2 * z1 + z3
        # 第三层神经网络
        v1 = u1-u3
        v2 = torch.sin(-u2)
        v3 = u1 * u3
        # 计算y1/y2
        y1 = v1 ** 2 + v2 ** 2
        y2 = v2 * v3
        return y1, y2


x = torch.tensor([1, 2, 0],dtype=torch.float, requires_grad=True)
model = Net()

y1, y2 = model(x)

# 分别对 y1 和 y2 执行反向传播
y1.backward(retain_graph=True)
print(x.grad)
x.grad.zero_()  # 清空梯度
y2.backward()

# 查看输入 x 的梯度
print(x.grad)