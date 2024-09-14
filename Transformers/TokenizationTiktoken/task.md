In this task, we want to perform the same steps as in previous one, but instead of tokenization at the character level,
we will use the
ready-made **cl100k_base** tokenizer from the tiktoken library. You can find out how to use this library at this
[link](https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb).

After the tokens are assigned numbers, you will need to assign an ordinal number starting from zero to all the tokens
that occur. That is, now the stoi and itos dictionaries will be responsible for translating from token numbers in
cl100k_base to an ordinal number and vice versa, respectively.

Also in tokenization function you need to return also a list of tokens (in ordinal system) which represents given text.

