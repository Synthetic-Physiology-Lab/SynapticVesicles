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

with open("input_lysosome.yml", "r") as f:
    p_init = yaml.full_load(f)

p = p_init["p"]
init = p_init["init"]

##################################################################################################
########################################## FIGURE 4 ##############################################
##################################################################################################

fig, ax = plt.subplots(2, 2)
C_init = [i for i in range(1, 202, 2)]

####### Luminal K+
P = p.copy()
P["P_Na"] = 0
P["P_Cl"] = 0
P["N_ClC"] = 0

INIT = init.copy()

pH_SS = list()
psi_SS = list()

for C in C_init:
    INIT["K_L"] = C * 1e-3

    PP, y0 = models.set_lysosome_model(P.copy(), INIT.copy())

    y = spi.odeint(models.lysosome_model, y0, t, args=(PP,))

    sol = models.extract_solution_lysosome(y, PP)

    psi, psi_tot = models.calculate_psi_lysosome(sol, PP)

    pH_SS.append(sol["pH"][-1])
    psi_SS.append(psi_tot[-1] * 1e3)

ax[0][0].plot(C_init, pH_SS)
ax[0][1].plot(C_init, psi_SS)

####### Luminal Na+
P = p.copy()
P["P_K"] = 0
P["P_Cl"] = 0
P["N_ClC"] = 0

INIT = init.copy()

pH_SS = list()
psi_SS = list()

for C in C_init:
    INIT["Na_L"] = C * 1e-3

    PP, y0 = models.set_lysosome_model(P.copy(), INIT.copy())

    y = spi.odeint(models.lysosome_model, y0, t, args=(PP,))

    sol = models.extract_solution_lysosome(y, PP)

    psi, psi_tot = models.calculate_psi_lysosome(sol, PP)

    pH_SS.append(sol["pH"][-1])
    psi_SS.append(psi_tot[-1] * 1e3)

ax[0][0].plot(C_init, pH_SS)
ax[0][1].plot(C_init, psi_SS)

####### Luminal Cl-
P = p.copy()
P["P_K"] = 0
P["P_Na"] = 0
P["N_ClC"] = 0

INIT = init.copy()

pH_SS = list()
psi_SS = list()

for C in C_init:
    INIT["Cl_L"] = C * 1e-3

    PP, y0 = models.set_lysosome_model(P.copy(), INIT.copy())

    y = spi.odeint(models.lysosome_model, y0, t, args=(PP,))

    sol = models.extract_solution_lysosome(y, PP)

    psi, psi_tot = models.calculate_psi_lysosome(sol, PP)

    pH_SS.append(sol["pH"][-1])
    psi_SS.append(psi_tot[-1] * 1e3)

ax[0][0].plot(C_init, pH_SS)
ax[0][1].plot(C_init, psi_SS)

####### Luminal Cl- ClC-7
P = p.copy()
P["P_K"] = 0
P["P_Na"] = 0
P["P_Cl"] = 0

INIT = init.copy()

pH_SS = list()
psi_SS = list()

for C in C_init:
    INIT["Cl_L"] = C * 1e-3

    PP, y0 = models.set_lysosome_model(P.copy(), INIT.copy())

    y = spi.odeint(models.lysosome_model, y0, t, args=(PP,))

    sol = models.extract_solution_lysosome(y, PP)

    psi, psi_tot = models.calculate_psi_lysosome(sol, PP)

    pH_SS.append(sol["pH"][-1])
    psi_SS.append(psi_tot[-1] * 1e3)

ax[0][0].plot(C_init, pH_SS)
ax[0][1].plot(C_init, psi_SS)

ax[0][0].set_xlim(0, 200)
ax[0][1].set_xlim(0, 200)
ax[0][0].set_ylim(3, 7.5)
ax[0][1].set_ylim(-40, 120)
ax[0][0].set_xlabel("Initial [K+]_L, [Na+]_L or [Cl-]_L [mM]")
ax[0][1].set_xlabel("Initial [K+]_L, [Na+]_L or [Cl-]_L [mM]")
ax[0][0].set_ylabel("Final pH_L")
ax[0][1].set_ylabel("Final Membrane Potential [mV]")
ax[0][0].legend(["K+ channel", "Na+ channel", "Cl- channel", "ClC-7 antiporter"])
ax[0][1].legend(["K+ channel", "Na+ channel", "Cl- channel", "ClC-7 antiporter"])


C_init = [i / 10 for i in range(1, 502, 5)]

####### Cytosolic Cl-
P = p.copy()
P["P_K"] = 0
P["P_Na"] = 0
P["N_ClC"] = 0

INIT = init.copy()

pH_SS = list()
psi_SS = list()

for C in C_init:
    P["Cl_C"] = C * 1e-3

    PP, y0 = models.set_lysosome_model(P.copy(), INIT.copy())

    y = spi.odeint(models.lysosome_model, y0, t, args=(PP,))

    sol = models.extract_solution_lysosome(y, PP)

    psi, psi_tot = models.calculate_psi_lysosome(sol, PP)

    pH_SS.append(sol["pH"][-1])
    psi_SS.append(psi_tot[-1] * 1e3)

ax[1][0].plot(C_init, pH_SS)
ax[1][1].plot(C_init, psi_SS)

####### Cytosolic Cl- ClC-7
P = p.copy()
P["P_K"] = 0
P["P_Na"] = 0
P["P_Cl"] = 0

INIT = init.copy()

pH_SS = list()
psi_SS = list()

for C in C_init:
    P["Cl_C"] = C * 1e-3

    PP, y0 = models.set_lysosome_model(P.copy(), INIT.copy())

    y = spi.odeint(models.lysosome_model, y0, t, args=(PP,))

    sol = models.extract_solution_lysosome(y, PP)

    psi, psi_tot = models.calculate_psi_lysosome(sol, PP)

    pH_SS.append(sol["pH"][-1])
    psi_SS.append(psi_tot[-1] * 1e3)

ax[1][0].plot(C_init, pH_SS)
ax[1][1].plot(C_init, psi_SS)

ax[1][0].set_xlim(0, 50)
ax[1][1].set_xlim(0, 50)
ax[1][0].set_ylim(4, 7)
ax[1][1].set_ylim(-50, 200)
ax[1][0].set_xlabel("Initial cytosolic [Cl-] [mM]")
ax[1][1].set_xlabel("Initial cytosolic [Cl-] [mM]")
ax[1][0].set_ylabel("Final pH_L")
ax[1][1].set_ylabel("Final Membrane Potential [mV]")
ax[1][0].legend(["Cl- channel", "ClC-7 antiporter"])
ax[1][1].legend(["Cl- channel", "ClC-7 antiporter"])
plt.show()
