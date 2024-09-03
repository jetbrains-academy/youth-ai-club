In this task you need to count backprop for variables from BatchNorm and Non-linearly layers
```
# BatchNorm layer
bnmeani = 1 / n * hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff ** 2
bnvar = 1/(n-1) * (bndiff2).sum(0, keepdim=True) 
bnvar_inv = (bnvar + 1e-5) ** (-0.5)
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias

# Non-linearity
h = torch.tanh(hpreact) # hidden layer

# Linear layer 2
logits = h @ W2 + b2
```