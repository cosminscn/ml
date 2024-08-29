import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.w1 = nn.Linear(2, 5)   # w1: (2, 5), b1: (5,)
        self.w2 = nn.Linear(5, 3)   # w2: (5, 3), b2: (3,)
        self.w3 = nn.Linear(3, 1)   # w3: (3, 1), b3: (1,)
        nn.init.constant_(self.w1.bias, 0)
        nn.init.constant_(self.w2.bias, 0)
        nn.init.constant_(self.w3.bias, 0)

    def forward(self, x, y):  # x: (B, 2), y: (B,)
        self.a1 = torch.tanh(self.w1(x))     # a1: (B, 5)
        self.a2 = torch.tanh(self.w2(self.a1))  # a2: (B, 3)
        self.a3 = torch.tanh(self.w3(self.a2))  # a3: (B, 1)
        loss = F.mse_loss(self.a3, y[:, None])  # loss: scalar
        return self.a3, loss

# PyTorch forward and backward pass
net = MLPNet()
X = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float32)  # X: (3, 2)
y = torch.tensor([0, 1, 1], dtype=torch.float32)  # y: (3,)

net.zero_grad()
y_hat, loss = net(X, y)
loss.backward()

# Extract gradients from PyTorch
grad_w1 = net.w1.weight.grad  # (5, 2)
grad_b1 = net.w1.bias.grad    # (5,)
grad_w2 = net.w2.weight.grad  # (3, 5)
grad_b2 = net.w2.bias.grad    # (3,)
grad_w3 = net.w3.weight.grad  # (1, 3)
grad_b3 = net.w3.bias.grad    # (1,)

# Manual forward pass
w1 = net.w1.weight.clone().detach().T  # (2, 5)
b1 = net.w1.bias.clone().detach()      # (5,)
w2 = net.w2.weight.clone().detach().T  # (5, 3)
b2 = net.w2.bias.clone().detach()      # (3,)
w3 = net.w3.weight.clone().detach().T  # (3, 1)
b3 = net.w3.bias.clone().detach()      # (1,)

def fwd(X, w1, w2, w3, b1, b2, b3, y):
    # X: (B, 2), w1: (2, 5), b1: (5,)
    a1 = torch.tanh(torch.mm(X, w1) + b1)   # (B, 5) = tanh((B, 2) @ (2, 5) + (5,))

    # a1: (B, 5), w2: (5, 3), b2: (3,)
    a2 = torch.tanh(torch.mm(a1, w2) + b2)  # (B, 3) = tanh((B, 5) @ (5, 3) + (3,))

    # a2: (B, 3), w3: (3, 1), b3: (1,)
    a3 = torch.tanh(torch.mm(a2, w3) + b3)  # (B, 1) = tanh((B, 3) @ (3, 1) + (1,))

    # y: (B,), a3: (B, 1)
    loss_manual = torch.mean((y[:, None] - a3) ** 2)  # scalar
    return a3, (a1, a2, a3), loss_manual

y_hat_manual, (a1, a2, a3), loss_manual = fwd(X, w1, w2, w3, b1, b2, b3, y)

# Manual backward pass
def loss_backward(a3, y):
    # a3: (B, 1), y: (B,)
    return 2 * (a3 - y[:, None]) / y.size(0)  # (B, 1)

def layer_backward(da_curr, w_curr, a_curr, a_prev):
    # da_curr: (B, h_curr), a_curr: (B, h_curr)
    dz_curr = da_curr * (1 - a_curr ** 2)  # (B, h_curr)

    # a_prev: (B, h_prev), dz_curr: (B, h_curr)
    dw_curr = a_prev.T.mm(dz_curr)  # (h_prev, h_curr) = (h_prev, B) @ (B, h_curr)
    db_curr = dz_curr.sum(dim=0)    # (h_curr,)

    # dz_curr: (B, h_curr), w_curr: (h_prev, h_curr)
    da_prev = dz_curr.mm(w_curr.T)  # (B, h_prev) = (B, h_curr) @ (h_curr, h_prev)

    return dw_curr, db_curr, da_prev

def backward_mse_3layers(X, a1, a2, a3, y, w1, w2, w3):
    # a3: (B, 1), y: (B,)
    da3 = loss_backward(a3, y)  # (B, 1)

    # da3: (B, 1), w3: (3, 1), a3: (B, 1), a2: (B, 3)
    dw3, db3, da2 = layer_backward(da3, w3, a3, a2)  # dw3: (3, 1), db3: (1,), da2: (B, 3)

    # da2: (B, 3), w2: (5, 3), a2: (B, 3), a1: (B, 5)
    dw2, db2, da1 = layer_backward(da2, w2, a2, a1)  # dw2: (5, 3), db2: (3,), da1: (B, 5)

    # da1: (B, 5), w1: (2, 5), a1: (B, 5), X: (B, 2)
    dw1, db1, _   = layer_backward(da1, w1, a1, X)   # dw1: (2, 5), db1: (5,), _: not used

    return dw1, db1, dw2, db2, dw3, db3

dw1, db1, dw2, db2, dw3, db3 = backward_mse_3layers(X, a1, a2, a3, y, w1, w2, w3)

# Compare results
def compare_tensors(tensor1, tensor2, name):
    if not torch.allclose(tensor1, tensor2, atol=1e-6):
        print(f"{name} mismatch")
        print(f"PyTorch: {tensor1}")
        print(f"Manual: {tensor2}")
    else:
        print(f"{name} match")

print("\nCOMPARISONS")
compare_tensors(y_hat, y_hat_manual, "y_hat")
compare_tensors(loss, loss_manual, "loss")
compare_tensors(grad_w1, dw1.T, "dw1")
compare_tensors(grad_b1, db1, "db1")
compare_tensors(grad_w2, dw2.T, "dw2")
compare_tensors(grad_b2, db2, "db2")
compare_tensors(grad_w3, dw3.T, "dw3")
compare_tensors(grad_b3, db3, "db3")

# h1 = Wh0
# dL/dW = dL/dh1 x dh1/dW

# W [mxn]
# h0 [n,1]
# h1 [m,1]

# J = [m, [mxn]]

# dh1_i
# ------
# dWk,p

# dL/dW
