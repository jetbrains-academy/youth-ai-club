from typing import Tuple

import torch


def count_grad(input_value: float) -> Tuple[float, float]:
    t_input = torch.tensor([input_value], requires_grad=True)
    output1 = torch.clip(t_input.exp(), 0, 1)
    output1.backward()
    result1 = t_input.grad.item()

    t_input.grad = None
    output2 = torch.clip(t_input ** 2, 0, 1)
    output2.backward()
    result2 = t_input.grad.item()

    return result1, result2
