import numpy as np
import sys
sys.path.append('.')
from tqdm import tqdm
import matplotlib.pyplot as plt
from scripts import ising
from scripts import torch_ising, torch_ising_3d
from multiprocessing import Pool

# Insight from using full energy
np.seterr(over='raise')

# ## Simple first run
#
# 1. Iniatilize NxN grid
# 2. Compute the initial energy $E$
# 3. Flip random spin and compute $\Delta E$
# 4. Accept or reject based on criteria
# 5. Continue to 3

N = 30
grid = ising.get_random_grid(N)

# ## Use pytorch convolution

def simulate_torch_3d(T,grid, J,mu,N, therm_sweeps=1500, measure_sweeps=800):
    beta = 1/T

    # Thermalise    
    conv = torch_ising_3d.get_conv_nn(J, mu, device='cuda')
    for ix in range(9*3*therm_sweeps):
        grid, dE, dM = torch_ising_3d.metrop_step(grid, conv, beta)
    
    E = [ising.ising_energy(grid[0][0], J, mu).cpu().numpy()]
    M = [grid.sum().cpu().numpy()]
    #grid = grid.cpu().numpy()[0,0]
    for ix in range(9*3*measure_sweeps):
        #dE = ising.metrop_step(grid, ix, J, mu, beta, N)
        grid, dE, dM = torch_ising_3d.metrop_step(grid, conv, beta)
        dE = dE or 0
        E.append( E[-1] + dE )
        M.append( M[-1] + dM )
    return E, M, grid

def simulate_torch_seq(temps, grid, J, mu, N, therm_sweeps=1500, measure_sweeps=800):
    em = []
    for T in tqdm(temps):
        a= (T, grid, J,mu,N, therm_sweeps, measure_sweeps)
        E, M, grid = simulate_torch_3d(*a)
        em.append((E,M))
    return em

# +
J = 0.5
mu = 0.

temps = np.linspace(0.05, 3, 20)
temps = np.concatenate((temps, np.linspace(0.65, 1.4, 40)))
temps = np.sort(temps)

grid = torch_ising_3d.get_random_grid(N, device='cuda')
pool = Pool(processes=2)
therm_sweeps = 800
measure_sweeps = 600

args = [(T, grid, J,mu,N,therm_sweeps, measure_sweeps) for T in temps]
# To run in parallel, use s-tarmap
#ems = pool.starmap(simulate_torch, args)
ems = simulate_torch_seq(temps, grid, J, mu, N, therm_sweeps, measure_sweeps)
eneg_tm, mag_tm = zip(*ems)
# -

# Clear tqdm
for instance in list(tqdm._instances):
    tqdm._decr_instances(instance)

# ## Save data for later analysis and plotting

# +
energies = np.mean(eneg_tm, axis=1)
heat = np.std(eneg_tm, axis=1)
magnetizations = np.mean(mag_tm, axis=1)
susc = np.std(mag_tm, axis=1)

exp = {
    'N':N
    ,'J':J
    ,'mu':mu
    ,'therm_sweeps':therm_sweeps
    ,'measure_sweeps':measure_sweeps
    ,'temps': temps
    ,'energies':energies
    ,'magn':magnetizations
    ,'heat':heat
    ,'susc':susc
}

fn = f'../data/exp_3d_N{N}_sweep{therm_sweeps}_mu{mu}_temps'
print('o:',fn)
np.save(fn, exp)
# -

