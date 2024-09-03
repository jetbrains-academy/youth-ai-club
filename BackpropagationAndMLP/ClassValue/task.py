import math


class Value:

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(label={self.label}, data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,), f'**{other}')
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), 'tanh')
        return out

    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self,), 'exp')
        return out

    # Now we need to implement subtraction and division: inverse operations for addition and multiplication

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * other ** -1

    # Now our program works for construction (a-2), but will it work for (2-a)? You can check this
    # Let's define the methods below to make everything work correctly
    def __radd__(self, other):  # other + self
        return self + other

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * self ** -1


# You can check the operation of the class using similar examples
a = Value(2.0)
b = Value(-3.0)
print((b / a).data)
