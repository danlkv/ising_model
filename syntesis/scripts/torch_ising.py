import torch as T
import numpy as np
from itertools import product
from torch.functional import F
import scripts.ising as ising

def get_nn_mask(J, mu):
     return np.array([
         [0, J, 0]
        ,[J, mu, J]
        ,[0, J, 0]
    ])
def get_funny_mask(J, mu):
     return np.array([
         [J, J, J/2]
        ,[J, mu, J]
        ,[J/2, J, -J/2]
    ])


def get_conv_torch(mask):
    """ Get 2d torch convolution with mask """
    in_chan, out_chan = 1, 1
    shape = mask.shape
    l = T.nn.Conv2d(in_chan, out_chan, shape
                        , stride=shape
                        , padding=0
                        , bias = False
                       )
    l.weight.data = T.from_numpy(mask[np.newaxis, np.newaxis, ...])
    return l

def grid_torch(grid):
    """ Convert 2-d numpy to 4-d torch with shape 1,1,N,N """
    gpu_grid = T.from_numpy(grid[np.newaxis, np.newaxis,...]).double()
    return gpu_grid


def get_conv_nn(J, mu, device='cuda'):
    nn_mask = get_nn_mask(J, mu)
    conv = get_conv_torch(nn_mask)
    return conv.to(device)

def get_random_grid(N, device='cuda'):
    g_ = grid_torch(ising.get_random_grid(N))
    return g_.to(device)

def metrop_step(grid, conv, beta):
    rix = np.random.randint(0, high=3, size=2)
    grid = T.roll(grid, shifts=tuple(rix), dims=(2,3))
    
    dE = 2*conv(grid)[0,0]
    
    scatter_ixs = [np.arange(1, d-1, 3) for d in grid.shape[2:]]
    ixs = (0,0) + np.ix_(*scatter_ixs)
    sub = grid[ixs]
    dE = sub*dE
    
    acc_prob = T.exp(-beta*F.relu(dE))
    random = T.rand_like(acc_prob)
    sub[acc_prob > random] *= -1
    grid[ixs] = sub
    dE[acc_prob < random] *= 0
    sub[acc_prob < random] *= 0
    return grid, float(dE.sum().detach()), 2*float(sub.sum().detach())
