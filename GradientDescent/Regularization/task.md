**Task 5.** Adding regularization

Implement the AdaGrad algorithm with $L_2$ regularization term $\|w\|^2$ in $Q(w_{k})$. 
The update rule is as follows:

$$
    G_0 = 0, \\
    G_{k + 1} = G_{k} + \left(\nabla_{w} Q(w_{k})\right)^2, \\
    w_{k + 1} = w_{k} - \dfrac{\eta_k}{\sqrt{\varepsilon + G_{k + 1}}} \nabla_{w} Q(w_{k})     
$$

Note that here $G_{k}$ is not a scalar but a vector.

Recall that regularization is an addition to the loss function 
that penalizes the norm of the weights. 
We will use $L_2$ regularization, so the loss function takes the following form:

$$
    Q(w) = \dfrac{1}{\ell} \sum\limits_{i=1}^{\ell} (a_w(x_i) - y_i)^2 + \dfrac{\mu}{2} \| w \|^2
$$
