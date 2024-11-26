In this assignment, you need to implement method `.fit()` for linear regression model
using various gradient descent options, 
and understand how to select hyperparameters for these methods.

**Task 1. GradientDescent**:

$$ w_{k + 1} = w_{k} - \eta_{k} \nabla_{w} Q(w_{k}) $$

**Task 2. StochasticDescent**:

$$ w_{k + 1} = w_{k} - \eta_{k} \nabla_{w} q_{i_{k}}(w_{k}) $$ 

Here, $\nabla_{w} q_{i_{k}}(w_{k})$ is the gradient estimate 
based on a batch of randomly selected objects.

**Task 3. MomentumDescent**:

$$
    h_0 = 0, \\
    h_{k + 1} = \alpha h_{k} + \eta_k \nabla_{w} Q(w_{k}), \\
    w_{k + 1} = w_{k} - h_{k + 1}
$$

**Task 4.** Adaptive Gradient Algorithm **Adagrad**:

$$
    G_0 = 0, \\
    G_{k + 1} = G_{k} + \left(\nabla_{w} Q(w_{k})\right)^2, \\
    w_{k + 1} = w_{k} - \dfrac{\eta_k}{\sqrt{\varepsilon + G_{k + 1}}} \nabla_{w} Q(w_{k})     
$$

Note that here $G_{k}$ is not a scalar but a vector.

In all the methods mentioned above, we are using the following formula for the step size:

$$
    \eta_{k} = \lambda \left(\dfrac{s_0}{s_0 + k}\right)^p
$$

In practice, it is sufficient to tune the parameter $\lambda$, 
while the other parameters can be set to default values: 
$ s_0 = 1, p = 0.5.$

We will use the Mean Squared Error (MSE) loss function:

$$
    Q(w) = \dfrac{1}{\ell} \sum\limits_{i=1}^{\ell} \left(a_w(x_i) - y_i\right)^2
$$

All calculations must be vectorized.
