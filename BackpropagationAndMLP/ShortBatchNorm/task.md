### Backpropagation through batchnorm
In previous task you've implemented backprop through all batchnorm's variables. 
Now try to shorten backward pass through batchnorm, i.e calculate **dhprebn** given  **dhpreact**.
It would be great if all calculations were in one line.
What the forward pass looks like:
```
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True) 
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
```
Or in a more concise form:
```
hpreact_fast = bngain * (hprebn - hprebn.mean(0, keepdim=True)) / torch.sqrt(hprebn.var(0, keepdim=True, unbiased=True) + 1e-5) + bnbias
```