In the next few tasks, you will need to implement the attention mechanism. In this task, you need to write a Head class. It should have the following methods:
1) *init(self, head_size)* - initialize the head.
2) *forward(self, x)* - forward pass of the head. x is a tensor of shape (batch_size, block_size, n_embd). The output should be a tensor of shape (batch_size, block_size, n_embd). 

