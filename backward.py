import matplotlib.pyplot as plt
import torch
x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = torch.Tensor([1.0])
#这里是要保存w后续经过backward函数计算的梯度值
w.requires_grad = True

def forward(x,w):
    return x*w

def loss(x,y,w):
    y_pred = forward(x,w)
    return (y_pred - y) ** 2

print("predict (before training)",4,forward(4,w))

for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l = loss(x,y,w)
        #通过链式法则来计算梯度
        l.backward()
        w.data = w.data - 0.01*w.grad.data
        w.grad.data.zero_()
        print("progress:",epoch,l.item())
print("predict (after training)",4,forward(4,w).item())