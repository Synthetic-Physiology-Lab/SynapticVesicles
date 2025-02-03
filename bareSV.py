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

with open("input_bareSV.yml", "r") as f:
    p_init = yaml.full_load(f)

p = p_init["p"]
init = p_init["init"]

##################################################################################################
########################################## FIGURE 2d #############################################
##################################################################################################

fig, ax = plt.subplots()

H_P = [i * 1e-5 for i in range(6, 601)]

P = p.copy()

INIT = init.copy()

pH_SS = list()

for h_p in H_P:
    P["P_H"] = h_p

    PP, y0 = models.set_bareSV_model(P, INIT)

    y = spi.odeint(models.bareSV_model, y0, t, args=(PP,))

    sol = models.extract_solution_bareSV(y, p)

    pH_SS.append(sol['pH'][-1])

ax.plot(H_P, pH_SS)
ax.set_xlim(6e-5, 6e-3)
ax.set_ylim(4.5, 7.5)
ax.set_xlabel("H+ permeability [cm/s]")
ax.set_ylabel("Final pH_L []")
plt.show()


