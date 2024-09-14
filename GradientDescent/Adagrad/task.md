**Task 4.** Adaptive Gradient Algorithm **Adagrad**

$$
    G_0 = 0, \\
    G_{k + 1} = G_{k} + \left(\nabla_{w} Q(w_{k})\right)^2, \\
    w_{k + 1} = w_{k} - \dfrac{\eta_k}{\sqrt{\varepsilon + G_{k + 1}}} \nabla_{w} Q(w_{k})     
$$

Note that here \( G_{k} \) is not a scalar but a vector.
