In this task, you need to implement your own version 
of linear regression trained using gradient descent, 
based on the templates provided in the `task.py` file — **LinearRegression**.

We recommend following these guidelines:
* All calculations should be vectorized
* Loops in `Python` are allowed only for the iterations of gradient descent
* The stopping criterion should use both of the following conditions (simultaneously):
    * The squared Euclidean norm of the difference between the weights on two consecutive iterations is less than `tolerance`
    * The maximum number of iterations `max_iter` is reached
* To monitor the convergence of the optimization process, 
we will use a list called `loss_history`, where the values of the loss function 
will be stored after each step, starting from the initial step (before the first step in the direction of the gradient)
