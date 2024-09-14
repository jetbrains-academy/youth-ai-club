Up to this point we have studied manual backpropagation. However, now we want to automate this process and build the first neural network prototype. 

Neural networks will be large mathematical expressions, so we need a data structure that maintain these expressions. 

We are going to create a class `Value` which will serve as a wrapper for all real numbers.

The class `Value` must:
 - Have a constructor that accepts float numbers as input
 - Be able to perform all basic arithmetic operations with its instances
 - Be able to perform all basic arithmetic operations between instances and float numbers

In the example you can see that the class `Value` has several fields.

The **data** field contains the value of a given number. The **label** field will contain the node name. 

The **repr** function is responsible for displaying information about a class object. 

As an example, you are given an implementation of the addition function. Let's look at it in more detail:

The return value is an instance of the class `Value`. 
It contains the result of the addition in the **data** field,
operation name in the **op** field and a list of terms from which this value was obtained
in the field **children**.
