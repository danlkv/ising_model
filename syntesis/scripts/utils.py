import numpy as np

def adjacent_indices_torus(idx, N):
    i, j = idx
    return [
        ((i+1)%N, j),
        ((i-1), j),
        (i, (j+1)%N),
        (i, (j-1)),
    ]

def ising_energy(grid, J, mu):
    N = len(grid)
    E = 0
    # This is not efficient, but is clear to read
    for i in range(N):
        for j in range(N):
            adj_ = adjacent_indices_torus((i,j), N)
            E += -J*grid[i,j]*sum(grid[ix] for ix in adj_)
            E += -mu*grid[i, j]
    return E

def get_random_grid(N):
    grid = np.random.randint(low=0, high=2, size=(N, N))
    grid = -1 + 2*grid
    return grid

#@profile
def metrop_step(grid, idx, J, mu, beta, N):
        i, j = idx
        x = grid[i,j]
        adj_ = adjacent_indices_torus((i,j), N)
        dE = J*x*sum(grid[ix] for ix in adj_) + mu*x
        dE *= 2

        if dE < 0:
            grid[i, j] = - x
            return dE

        accept_p = np.exp(-beta*dE)
        if accept_p>np.random.rand():
            grid[i, j] = - x
            return dE
