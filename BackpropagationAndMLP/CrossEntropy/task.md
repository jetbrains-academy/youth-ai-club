### Backprop through cross_entropy
In previous parts you've implemented backprop for logits 
through all variables manually. Now you need to try to reduce the number of rows in the backward 
pass count. To complete this challenge look at the 
mathematical expression of the loss, take the 
derivative, simplify the expression, and just write 
it out.

#### Shortening the forward pass
before:
```
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes 
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum ** (-1) 
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()
```

now:
```
loss_fast = F.cross_entropy(logits, Yb)
```
It would be great if your solution consisted of no 
more than 3 lines :)