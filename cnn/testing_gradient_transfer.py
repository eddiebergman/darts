import torch
import numpy as np

if __name__ == "__main__":
    gpu_id = 0

    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    mat = torch.ones((2,2)).cuda()
