import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import models
from scipy.interpolate import RegularGridInterpolator
import lmfit


def pH_exp_decay(x):
    tau = 20
    y = 5.98 + (6.6 - 5.98) * np.exp(-x / tau)
    return y


def residual_bare(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["P_H"] = parvals["P_H"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model_constant, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    res = sol["pH"] - data
    return res


def residual_pH(pars, t, data):
    parvals = pars.valuesdict()
    tau = parvals["tau"]

    sol = pH_exp_decay(t, tau)

    res = sol - data
    return res


def GABA_exp_decay(x, p, init):
    p["N_VGLUT"] = 0
    p["N_VGAT"] = 0
    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model_constant, y0, t, args=(p,))
    sol_0 = models.extract_solution_SV(y, p)
    pH_0 = sol_0["pH"]

    tau = 25
    y = pH_0 + 0.42 * (1 - np.exp(-x / tau))
    return y


def residual_GABA(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["k_GABA"] = parvals["k_GABA"]
    p["tau_GABA"] = parvals["tau_GABA"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model_constant, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    res = sol["pH"] - data
    return res


def GLUT_exp_decay(x):
    tau = 25
    y = 5.8 + (6.6 - 5.8) * np.exp(-x / tau)
    return y


def residual_GLUT(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["k_GLUT"] = parvals["k_GLUT"]
    p["tau_GLUT"] = parvals["tau_GLUT"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model_constant, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    res = sol["pH"] - data
    return res


t_start = 0
t_stop = 2000
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

with open("input_SV.yml", "r") as f:
    p_init = yaml.full_load(f)

p = p_init["p"]
init = p_init["init"]

##################################################################################################
########################################## bare span #############################################
##################################################################################################

t_start = 0
t_stop = 2000
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

INIT = init.copy()
INIT["pH_L"] = 6.6
data = pH_exp_decay(t)
# plt.plot(t, data)
# plt.show()

INIT = init.copy()
INIT["pH_L"] = 6.6
P = p.copy()
P["N_VGLUT"] = 0
P["N_VGAT"] = 0

params = lmfit.Parameters()
params.add("P_H", value=0.1, min=0)

fit_result = lmfit.minimize(
    residual_bare, params, args=(t,), kws={"data": data, "p": P, "init": INIT}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P_H_ref = parvals["P_H"]
P["P_H"] = P_H_ref

PP, y0 = models.set_SV_model(P, INIT)
y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, _ = models.calculate_psi_SV(sol, PP)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, sol["GABA"])
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("GABA molecules []")
ax[1].plot(t, sol["pH"])
ax[1].plot(t, data)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("pH_L []")
ax[2].plot(t, psi * 1e3)
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("psi [mV]")
plt.tight_layout()
plt.show()

##################################################################################################
########################################## GABA span #############################################
##################################################################################################

t_start = 0
t_stop = 2000
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

INIT = init.copy()
INIT["pH_L"] = 6.6
P = p.copy()
P["P_H"] = P_H_ref
data = GABA_exp_decay(t, P.copy(), INIT.copy())
# plt.plot(t, data)
# plt.show()

INIT = init.copy()
INIT["pH_L"] = 6.6
P = p.copy()
P["N_VGLUT"] = 0

params = lmfit.Parameters()
params.add("k_GABA", value=50, min=0, max=75)
params.add("tau_GABA", value=25, min=0)

fit_result = lmfit.minimize(
    residual_GABA, params, args=(t,), kws={"data": data, "p": P, "init": INIT}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P["k_GABA"] = parvals["k_GABA"]
P["tau_GABA"] = parvals["tau_GABA"]

PP, y0 = models.set_SV_model(P, INIT)
y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, _ = models.calculate_psi_SV(sol, PP)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, sol["GABA"])
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("GABA molecules []")
ax[1].plot(t, sol["pH"])
ax[1].plot(t, data)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("pH_L []")
ax[2].plot(t, psi * 1e3)
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("psi [mV]")
plt.tight_layout()
plt.show()

##################################################################################################
########################################## GLUT span #############################################
##################################################################################################

t_start = 0
t_stop = 2000
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

data = GLUT_exp_decay(t)

INIT = init.copy()
INIT["pH_L"] = 6.6
P = p.copy()
P["N_VGAT"] = 0

params = lmfit.Parameters()
params.add("k_GLUT", value=7, min=0)
params.add("tau_GLUT", value=1, min=0)

fit_result = lmfit.minimize(
    residual_GLUT, params, args=(t,), kws={"data": data, "p": P, "init": INIT}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P["k_GLUT"] = parvals["k_GLUT"]
P["tau_GLUT"] = parvals["tau_GLUT"]

PP, y0 = models.set_SV_model(P, INIT)
y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, _ = models.calculate_psi_SV(sol, PP)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, sol["GLUT"])
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("GLUT molecules []")
ax[1].plot(t, sol["pH"])
ax[1].plot(t, data)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("pH_L []")
ax[2].plot(t, psi * 1e3)
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("psi [mV]")
plt.tight_layout()
plt.show()

##################################################################################################
########################################## GABA span #############################################
##################################################################################################

fig, ax = plt.subplots(2, 1)

K = [i for i in range(1, 151)]

P = p.copy()
P["N_VGLUT"] = 0

INIT = init.copy()

pH_SS = list()
GABA_SS = list()

for k in K:
    P["k_GABA"] = k

    PP, y0 = models.set_SV_model(P, INIT)

    y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))

    sol = models.extract_solution_SV(y, PP)

    pH_SS.append(sol["pH"][-1])
    GABA_SS.append(sol["GABA"][-1])

K = np.array(K)
pH_SS = np.array(pH_SS)
pH_ref = 6.4
ind = np.nonzero(pH_SS < pH_ref)[0][-1]
K_ref = K[ind] + ((K[ind + 1] - K[ind]) / (pH_SS[ind + 1] - pH_SS[ind])) * (
    pH_ref - pH_SS[ind]
)
print(K_ref)

ax[0].plot(K, pH_SS)
# ax.plot([6e-5, P_H_ref, P_H_ref], [pH_ref, pH_ref, 4.5], color='k', linestyle='--')
# ax.set_xlim(6e-5, 6e-3)
# ax.set_ylim(4.5, 7.5)
ax[0].set_xlabel("GABA transport rate [s^-1]")
ax[0].set_ylabel("Final pH_L []")

ax[1].plot(K, GABA_SS)
# ax.plot([6e-5, P_H_ref, P_H_ref], [pH_ref, pH_ref, 4.5], color='k', linestyle='--')
# ax.set_xlim(6e-5, 6e-3)
# ax.set_ylim(4.5, 7.5)
ax[1].set_xlabel("GABA transport rate [s^-1]")
ax[1].set_ylabel("Final GABA molecules []")
plt.tight_layout()
plt.show()

##################################################################################################
########################################## GABA #############################################
##################################################################################################

fig, ax = plt.subplots(3, 1)

P = p.copy()
P["N_VGLUT"] = 0
P["k_GABA"] = K_ref

INIT = init.copy()

PP, y0 = models.set_SV_model(P, INIT)

y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))

sol = models.extract_solution_SV(y, PP)

psi, _ = models.calculate_psi_SV(sol, PP)

RTF = PP["R"] * (PP["T"] + 273.15) / PP["F"]
pHe = PP["pH_C"] + PP["psi_o"] / (RTF * 2.3)
pHi = sol["pH"] + PP["psi_i"] / (RTF * 2.3)
delta_u_H = psi * 1e3 + 2.3 * RTF * 1e3 * (pHe - pHi)

ax[0].plot(t, sol["GABA"])
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("GABA molecules []")
ax[1].plot(t, sol["pH"])
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("pH_L []")
ax[2].plot(t, psi * 1e3)
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("psi [mV]")
# ax[3].plot(t, delta_u_H)
plt.tight_layout()
plt.show()
