**Task 3. MomentumDescent**

Implement the Momentum Descent method. 
Modify the `Descent` class to support the 
Momentum Descent method. 
It is a modification of the Gradient Descent method 
that accelerates the convergence of the optimization 
algorithm. The Momentum Descent method uses the following update rule:

$$
    h_0 = 0, \\
    h_{k + 1} = \alpha h_{k} + \eta_k \nabla_{w} Q(w_{k}), \\
    w_{k + 1} = w_{k} - h_{k + 1}
$$
