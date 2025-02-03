import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import models
from scipy.interpolate import RegularGridInterpolator

t_start = 0
t_stop = 2000
dt = 0.002
t = np.arange(start=t_start, stop=t_stop, step=dt)

with open('input.yml', 'r') as f:
    p_init = yaml.full_load(f)

p = p_init['p']
init = p_init['init']

##################################################################################################
########################################## FIGURE 6 ##############################################
##################################################################################################

fig, ax = plt.subplots()
C_init = [i for i in range(2,203,8)]

####### Luminal K+
P = p.copy()
P['P_Na'] = 0
P['P_Cl'] = 0
P['N_ClC'] = 0
P['P_W'] = 0.052
P['d'] = 0

INIT = init.copy()

delta_V_perc_SS = list()

for C in C_init:
	INIT['K_L'] = C*1e-3
	
	PP, y0 = models.set_lysosome_model_MADONNA(P, INIT)

	y = spi.odeint(models.lysosome_model_MADONNA, y0, t, args=(PP,))
	
	sol = models.extract_solution(y, PP)
	
	delta_V_perc = (sol['V'][-1] - sol['V'][0]) / sol['V'][0] * 100
	
	delta_V_perc_SS.append(delta_V_perc)
	
ax.plot(C_init, delta_V_perc_SS)

####### Luminal Na+
P = p.copy()
P['P_K'] = 0
P['P_Cl'] = 0
P['N_ClC'] = 0
P['P_W'] = 0.052
P['d'] = 0

INIT = init.copy()

delta_V_perc_SS = list()

for C in C_init:
	INIT['Na_L'] = C*1e-3
	
	PP, y0 = models.set_lysosome_model_MADONNA(P, INIT)

	y = spi.odeint(models.lysosome_model_MADONNA, y0, t, args=(PP,))
	
	sol = models.extract_solution(y, PP)
	
	delta_V_perc = (sol['V'][-1] - sol['V'][0]) / sol['V'][0] * 100
	
	delta_V_perc_SS.append(delta_V_perc)
	
ax.plot(C_init, delta_V_perc_SS)

####### Luminal Cl-
P = p.copy()
P['P_K'] = 0
P['P_Na'] = 0
P['N_ClC'] = 0
P['P_W'] = 0.052
P['d'] = 0

INIT = init.copy()

delta_V_perc_SS = list()

for C in C_init:
	INIT['Cl_L'] = C*1e-3
	
	PP, y0 = models.set_lysosome_model_MADONNA(P, INIT)

	y = spi.odeint(models.lysosome_model_MADONNA, y0, t, args=(PP,))
	
	sol = models.extract_solution(y, PP)
	
	delta_V_perc = (sol['V'][-1] - sol['V'][0]) / sol['V'][0] * 100
	
	delta_V_perc_SS.append(delta_V_perc)
	
ax.plot(C_init, delta_V_perc_SS)

####### Luminal Cl- ClC-7
P = p.copy()
P['P_K'] = 0
P['P_Na'] = 0
P['P_Cl'] = 0
P['P_W'] = 0.052
P['d'] = 0

INIT = init.copy()

delta_V_perc_SS = list()

for C in C_init:
	INIT['Cl_L'] = C*1e-3
	
	PP, y0 = models.set_lysosome_model_MADONNA(P, INIT)

	y = spi.odeint(models.lysosome_model_MADONNA, y0, t, args=(PP,))
	
	sol = models.extract_solution(y, PP)
	
	delta_V_perc = (sol['V'][-1] - sol['V'][0]) / sol['V'][0] * 100
	
	delta_V_perc_SS.append(delta_V_perc)
	
ax.plot(C_init, delta_V_perc_SS)


ax.set_xlim(0, 200)
ax.set_ylim(-50, 50)
ax.set_xlabel('Initial [K+]_L, [Na+]_L or [Cl-]_L [mM]')
ax.set_ylabel('Change in the Final Volume [%]')
ax.legend(['K+ channel', 'Na+ channel', 'Cl- channel', 'ClC-7 antiporter'])
plt.show()


