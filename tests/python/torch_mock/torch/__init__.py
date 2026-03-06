import numpy as np

class Tensor:
    def __init__(self, data, device="cpu"):
        self.data = np.array(data, dtype=np.float32)
        self.device = device
        
    def to(self, device):
        self.device = device
        return self
        
    def __add__(self, other):
        return Tensor(self.data + other.data, self.device)
        
def randn(*size, device="cpu"):
    if len(size) == 1 and isinstance(size[0], tuple):
        size = size[0]
    return Tensor(np.random.randn(*size), device=device)

def matmul(A, B):
    return Tensor(np.dot(A.data, B.data), device=A.device)

class cuda:
    @staticmethod
    def is_available():
        return True
        
    @staticmethod
    def synchronize():
        pass
