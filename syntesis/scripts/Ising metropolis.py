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
# <div class="toc"><ul class="toc-item"><li><span><a href="#Simple-first-run" data-toc-modified-id="Simple-first-run-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Simple first run</a></span></li><li><span><a href="#Use-simple-serial-algorithm" data-toc-modified-id="Use-simple-serial-algorithm-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Use simple serial algorithm</a></span></li><li><span><a href="#Use-pytorch-convolution" data-toc-modified-id="Use-pytorch-convolution-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Use pytorch convolution</a></span></li><li><span><a href="#Save-data-for-later-analysis-and-plotting" data-toc-modified-id="Save-data-for-later-analysis-and-plotting-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Save data for later analysis and plotting</a></span></li></ul></div>

# +
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scripts import ising
from scripts import torch_ising

# %load_ext autoreload
# %autoreload 2

# Insight from using full energy
np.seterr(over='raise')


# -

# ## Simple first run
#
# 1. Iniatilize NxN grid
# 2. Compute the initial energy $E$
# 3. Flip random spin and compute $\Delta E$
# 4. Accept or reject based on criteria
# 5. Continue to 3

#@profile
def random_ix(N, steps):
    randix = (np.random.randint(0, N, size=(steps,2)))
    return randix


N = 50
grid = ising.get_random_grid(N)
#plt.imshow(grid)
# Rescale to +-1

# ## Use simple serial algorithm

# +
J = 0.5
mu = 0

temps = np.linspace(0.05, 3, 100)
eneg_tm = []
mag_tm = []
for T in temps:
    beta = 1/T
    grid = ising.get_random_grid(N)
    
    # Thermalise    
    therm_sweeps = 200
    for ix in random_ix(N, steps=N**2*therm_sweeps):
        dE = ising.metrop_step(grid, ix, J, mu, beta, N)
    
    E = [ising.ising_energy(grid, J, mu)]
    M = [np.mean(grid)]
    print('measure t=',T)
    for ix in random_ix(N, steps=N**2*100):
        raise
        dE = ising.metrop_step(grid, ix, J, mu, beta, N)
        dE = dE or 0
        E.append( E[-1] + dE )
        M.append(np.mean(grid))
        
    print('done measure')
    eneg_tm.append(E)
    mag_tm.append(M)


# +
fig, axs = plt.subplots(2,2, figsize=(8,6))

[ax.set_title(t) for ax, t in zip(sum(map(list, axs),[]),
                              ['Energy','Specific Heat', 'Magnetization', 'Susceptibility'])]

axs[0,0].plot(temps, list(map(np.mean, eneg_tm)))
axs[0,1].plot(temps, list(map(np.std, eneg_tm)))
axs[1,0].plot(temps, list(map(np.mean, mag_tm)))
axs[1,1].plot(temps, list(map(np.std, mag_tm)))

# -

# ## Use pytorch convolution

# +
J = 0.5
mu = 0

temps = np.linspace(0.05, 2, 30)
eneg_tm = []
mag_tm = []
grid = torch_ising.get_random_grid(N, device='cpu')
for T in tqdm(temps):
    beta = 1/T
    
    # Thermalise    
    therm_sweeps = 600
    conv = torch_ising.get_conv_nn(J, mu, device='cpu')
    for ix in random_ix(N, steps=9*therm_sweeps):
        grid, dE, dM = torch_ising.metrop_step(grid, conv, beta)
    
    measure_sweeps = 600
    E = [ising.ising_energy(grid[0][0], J, mu).cpu().numpy()]
    M = [grid.sum().cpu().numpy()]
    #grid = grid.cpu().numpy()[0,0]
    for ix in random_ix(N, steps=9*measure_sweeps):
        #dE = ising.metrop_step(grid, ix, J, mu, beta, N)
        grid, dE, dM = torch_ising.metrop_step(grid, conv, beta)
        dE = dE or 0
        E.append( E[-1] + dE )
        M.append( M[-1] + dM )
        
    eneg_tm.append(E)
    mag_tm.append(M)
# -






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

np.save(f'../data/exp_N{N}_sweep{therm_sweeps}', exp)
# -


plt.plot(mag_tm[20])

# smoothing
plt.plot(np.convolve(energies, np.ones((N**2,))/N**2, mode='valid'))

fig, axs = plt.subplots(2,1, figsize=(5,5))
plt.plot(averages, label='average occupation') 
plt.legend()
plt.sca(axs[0])
plt.plot(energies, label='Energy', color='orange') 
plt.legend()

#grid = get_random_grid(50)
plt.imshow(grid[0,0])
