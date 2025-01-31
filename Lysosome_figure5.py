import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import models
from scipy.interpolate import RegularGridInterpolator

t_start = 0
t_stop = 2000
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

with open('input.yml', 'r') as f:
    p_init = yaml.full_load(f)

p = p_init['p']
init = p_init['init']

##################################################################################################
########################################## FIGURE 5 ##############################################
##################################################################################################

fig, ax = plt.subplots()

####### K+ and Na+ channels
P = p.copy()
P['P_Cl'] = 0
P['N_ClC'] = 0

INIT = init.copy()
INIT['K_L'] = 50e-3
INIT['Na_L'] = 20e-3
INIT['Cl_L'] = 1e-3
INIT['pH_L'] = 6

PP, y0 = models.set_lysosome_model_MADONNA(P, INIT)

y = spi.odeint(models.lysosome_model_MADONNA, y0, t, args=(PP,))

sol = models.extract_solution(y, PP)

ax.plot(t, sol['pH'])

####### K+ and Na+ channels and ClC-7 antiporter
P = p.copy()
P['P_Cl'] = 0
P['N_ClC'] = 300

INIT = init.copy()
INIT['K_L'] = 50e-3
INIT['Na_L'] = 20e-3
INIT['Cl_L'] = 1e-3
INIT['pH_L'] = 6

PP, y0 = models.set_lysosome_model_MADONNA(P, INIT)

y = spi.odeint(models.lysosome_model_MADONNA, y0, t, args=(PP,))

sol = models.extract_solution(y, PP)

ax.plot(t, sol['pH'])

ax.set_xlim(0, 2000)
ax.set_ylim(4.5, 6)
ax.set_xlabel('Time [s]')
ax.set_ylabel('pH_L')
ax.legend(['K+ and Na+ channels', 'K+ and Na+ channels and ClC-7 antiporter'])
plt.show()



