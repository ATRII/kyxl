import numpy as np
import torch
import random
from torch.utils.tensorboard import SummaryWriter
# a = [([{1: [[1, 2], [2, 3]], 2:3}]*10, [1, 2]*10)]*10
# a0 = random.sample(a, 2)
# a1, a2 = map(np.asarray, zip(*a0))
# print(a1, a2)
# a3 = map(np.asarray, zip(*[[i[1] for i in j] for j in a1]))
# print([[i[1] for i in j] for j in a1])
# writer = SummaryWriter('./log')
# a = 0
# for i in range(10):
#     a += 1
#     writer.add_scalar("i", a,
#                       global_step=None, walltime=None)
#     a += 1
a = [[], []]
a[1].append(1)
print(a)
