**Tokenization** is the process of *splitting text into some pieces* - **tokens**. With different partitioning, there will be
different results of text generation, and therefore this process is very important. You can split words just character by
character, or you can split words into some semantic units (root, suffix, and so on).

After splitting the text into tokens, we need to represent each token as an **embedding vector**. This is necessary so that
the network can assign some characteristics to tokens and better combine them into text. At the beginning, these vectors
can be set randomly, this is a trainable parameter, so it will adjust during the training process. Usually, embedding
for all tokens is presented simply as a table, where the first row is a vector for the first token, the second row for
the second, and so on.