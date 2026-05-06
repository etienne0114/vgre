#!/usr/bin/env python3
import sys

try:
    import torch
    import torch.nn as nn
except Exception as e:
    print("torch is not installed or failed to import:", e)
    print("Install requirements: python3 -m venv .venv && . .venv/bin/activate && pip install -r tests/python/requirements.txt")
    sys.exit(2)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # small model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 1)
    )
    model.to(device)
    model.train()

    # batched inputs
    x = torch.randn(8, 10, device=device, requires_grad=True)

    y = model(x).sum()
    y.backward()

    # verify gradients exist and are finite
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f'Parameter {name} has no gradient')
            return 3
        if not torch.isfinite(param.grad).all():
            print(f'Parameter {name} gradient contains non-finite values')
            return 4

    if x.grad is None:
        print('Input tensor has no gradient')
        return 5
    if not torch.isfinite(x.grad).all():
        print('Input gradient contains non-finite values')
        return 6

    print('Gradient verification succeeded on', device)
    return 0

if __name__ == '__main__':
    sys.exit(main())
