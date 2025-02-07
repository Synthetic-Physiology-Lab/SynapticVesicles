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

with open("input_SV.yml", "r") as f:
    p_init = yaml.full_load(f)

p = p_init["p"]
init = p_init["init"]

##################################################################################################
########################################## GABA span #############################################
##################################################################################################

fig, ax = plt.subplots(2,1)

K = [i for i in range(1, 151)]

P = p.copy()
P["N_VGLUT"] = 0

INIT = init.copy()

pH_SS = list()
GABA_SS = list()

for k in K:
    P["k_GABA"] = k

    PP, y0 = models.set_SV_model(P, INIT)

    y = spi.odeint(models.SV_model, y0, t, args=(PP,))

    sol = models.extract_solution_SV(y, PP)

    pH_SS.append(sol["pH"][-1])
    GABA_SS.append(sol["GABA"][-1])

K = np.array(K)
pH_SS = np.array(pH_SS)
pH_ref = 6.4
ind = np.nonzero(pH_SS < pH_ref)[0][-1]
K_ref = K[ind] + ( (K[ind+1]-K[ind])/(pH_SS[ind+1]-pH_SS[ind]) ) * (pH_ref-pH_SS[ind])
print(K_ref)

ax[0].plot(K, pH_SS)
#ax.plot([6e-5, P_H_ref, P_H_ref], [pH_ref, pH_ref, 4.5], color='k', linestyle='--')
#ax.set_xlim(6e-5, 6e-3)
#ax.set_ylim(4.5, 7.5)
ax[0].set_xlabel("GABA transport rate [s^-1]")
ax[0].set_ylabel("Final pH_L []")

ax[1].plot(K, GABA_SS)
#ax.plot([6e-5, P_H_ref, P_H_ref], [pH_ref, pH_ref, 4.5], color='k', linestyle='--')
#ax.set_xlim(6e-5, 6e-3)
#ax.set_ylim(4.5, 7.5)
ax[1].set_xlabel("GABA transport rate [s^-1]")
ax[1].set_ylabel("Final GABA molecules []")
plt.tight_layout()
plt.show()

##################################################################################################
########################################## GABA #############################################
##################################################################################################

fig, ax = plt.subplots(3,1)

P = p.copy()
P["N_VGLUT"] = 0
P["k_GABA"] = K_ref

INIT = init.copy()

PP, y0 = models.set_SV_model(P, INIT)

y = spi.odeint(models.SV_model, y0, t, args=(PP,))

sol = models.extract_solution_SV(y, PP)

psi, _ = models.calculate_psi_SV(sol, PP)

RTF = PP["R"] * (PP["T"] + 273.15) / PP["F"]
pHe = PP["pH_C"] + PP["psi_o"] / (RTF * 2.3)
pHi = sol['pH'] + PP["psi_i"] / (RTF * 2.3)
delta_u_H = psi * 1e3 + 2.3 * RTF * 1e3 * (pHe - pHi)

ax[0].plot(t, sol['GABA'])
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("GABA molecules []")
ax[1].plot(t, sol['pH'])
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("pH_L []")
ax[2].plot(t, psi*1e3)
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("psi [mV]")
#ax[3].plot(t, delta_u_H)
plt.tight_layout()
plt.show()


##################################################################################################
########################################## GLUT span #############################################
##################################################################################################

fig, ax = plt.subplots(2,1)

#K = [i * 1e-1 for i in range(10, 151)]
K = [i for i in range(1, 16)]

P = p.copy()
P["N_VGAT"] = 0

INIT = init.copy()

pH_SS = list()
GLUT_SS = list()

for k in K:
    P["k_GLUT"] = k

    PP, y0 = models.set_SV_model(P, INIT)

    y = spi.odeint(models.SV_model, y0, t, args=(PP,))

    sol = models.extract_solution_SV(y, PP)

    pH_SS.append(sol["pH"][-1])
    GLUT_SS.append(sol["GLUT"][-1])

K = np.array(K)
pH_SS = np.array(pH_SS)
pH_ref = 5.8
ind = np.nonzero(pH_SS < pH_ref)[0][-1]
K_ref = K[ind] + ( (K[ind+1]-K[ind])/(pH_SS[ind+1]-pH_SS[ind]) ) * (pH_ref-pH_SS[ind])
print(K_ref)

ax[0].plot(K, pH_SS)
#ax.plot([6e-5, P_H_ref, P_H_ref], [pH_ref, pH_ref, 4.5], color='k', linestyle='--')
#ax.set_xlim(6e-5, 6e-3)
#ax.set_ylim(4.5, 7.5)
ax[0].set_xlabel("GLUT transport rate [s^-1]")
ax[0].set_ylabel("Final pH_L []")

ax[1].plot(K, GLUT_SS)
#ax.plot([6e-5, P_H_ref, P_H_ref], [pH_ref, pH_ref, 4.5], color='k', linestyle='--')
#ax.set_xlim(6e-5, 6e-3)
#ax.set_ylim(4.5, 7.5)
ax[1].set_xlabel("GLUT transport rate [s^-1]")
ax[1].set_ylabel("Final GLUT molecules []")
plt.tight_layout()
plt.show()

##################################################################################################
########################################## GLUT #############################################
##################################################################################################

fig, ax = plt.subplots(3,1)

P = p.copy()
P["N_VGAT"] = 0
P["k_GLUT"] = K_ref

INIT = init.copy()

PP, y0 = models.set_SV_model(P, INIT)

y = spi.odeint(models.SV_model, y0, t, args=(PP,))

sol = models.extract_solution_SV(y, PP)

psi, _ = models.calculate_psi_SV(sol, PP)

RTF = PP["R"] * (PP["T"] + 273.15) / PP["F"]
pHe = PP["pH_C"] + PP["psi_o"] / (RTF * 2.3)
pHi = sol['pH'] + PP["psi_i"] / (RTF * 2.3)
delta_u_H = psi * 1e3 + 2.3 * RTF * 1e3 * (pHe - pHi)

ax[0].plot(t, sol['GLUT'])
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("GLUT molecules []")
ax[1].plot(t, sol['pH'])
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("pH_L []")
ax[2].plot(t, psi*1e3)
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("psi [mV]")
#ax[3].plot(t, delta_u_H)
plt.tight_layout()
plt.show()







