# ---
# jupyter:
#   jupytext:
#     formats: ipynb,scripts//py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"></ul></div>
# -

import sys
import numpy as np
import pyrofiler as prof
import itertools

# +
S = int(40)**2

x = np.random.random(S)
print(x.dtype)


# +
def iter_nn_idx(i, shape):
    dim = len(i)
    assert dim == len(shape)
    deltas = itertools.product([-1, 0, 1], repeat=dim)
    return deltas

def test_iter_nn_idx():
    d = iter_nn_idx([2,2], (5,5))
    print(list(d))
test_iter_nn_idx()


# +
 
def get_nn_mask(i, side=int(1e3)):
    shape = (side, side)
    m = np.zeros((side, side))
    ix = np.unravel_index(i, shape)
    
    m[ix[0]+1, ix[1]] = 1
    m[ix[0]-1, ix[1]] = 1
    m[ix[0], ix[1]+1] = 1
    m[ix[0], ix[1]-1] = 1
    
    return m.flatten()
    
m = get_nn_mask(1+1000, side=int(np.sqrt(S)))
# -

# %%timeit
c = np.sum(x*m)

i = 5
idxs = [
    [i,i, i+1, i-1],
    [i+1, i-1, i,i]
]
side = int(np.sqrt(S))
ix = np.ravel_multi_index(idxs, (side, side))

# %%timeit
nonz = [x[i] for i in ix ]
sum(nonz)



d = {1:2}

# %%timeit
np.exp(4)

# %%timeit
d[1]

# %%timeit
x[1]
