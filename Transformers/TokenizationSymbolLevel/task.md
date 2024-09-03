In this task, you will need to write two functions:

1) **Tokenization function.** Tokenize at the character level the text given to you in the form of a list of words.

   You need to assign a unique number to each character found in the text (and if you look at the sorted list of
   characters,
   then the smallest character should have the number zero, and any other one should have one more than the previous
   one).

   Do not forget to add a special character ('#') indicating the beginning\end of the text, we believe that this
   symbol should have the smallest token (i.e. zero).

   The function should return the `stoi` dictionary - where its number is
   stored by symbol and the `itos` dictionary - where its symbol is stored by number
 
2) **Embeddings function.** You are given a number of unique tokens, dimensions of vector which represents a token and seed
   for generator. You need to create a matrix of embedding vectors for tokens. `C[i]` (tensor of length n_embd) vector
   representation of token `i`. Since embedding vectors are trainable, in this function you just need to create them
   randomly using the `randn` function and a generator with the seed given to you.