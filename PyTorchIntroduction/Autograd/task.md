In this task, you should learn how to use torch autograd tools.

You are given 2 functions $clip(\exp(input), 0, 1)$ and $clip(input^2, 0, 1)$, where
$$
clip(x, min, max) =
\begin{cases}
min, if \ x < min; \\
max, if \ x > max; \\
x, otherwise;
\end{cases}
$$

Return grads of both functions with respect to input.