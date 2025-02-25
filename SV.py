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

    res = sol["pH"][-1] - data[-1]
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
    # p["tau_GABA_2"] = parvals["tau_GABA_2"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model_constant, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    # molecules = 4000
    # data = np.append(data, molecules)
    # res = np.append(sol["pH"], sol["GABA"][-1]) - data
    res = sol["pH"] - data
    return res


def residual_GABA_2(pars, t, data, p, init):
    parvals = pars.valuesdict()
    # p["k_GABA"] = parvals["k_GABA"]
    # p["tau_GABA"] = parvals["tau_GABA"]
    p["tau_GABA_2"] = parvals["tau_GABA_2"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    molecules = 4000
    data = np.append(data, molecules)
    res = np.append(sol["pH"], sol["GABA"][-1]) - data
    # res = sol["pH"] - data
    return res


def GLUT_exp_decay(x):
    tau = 25
    y = 5.8 + (6.6 - 5.8) * np.exp(-x / tau)
    return y


def residual_GLUT(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["k_GLUT"] = parvals["k_GLUT"]
    # p["tau_GLUT"] = parvals["tau_GLUT"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model_constant, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    res = sol["pH"] - data
    return res


def residual_GLUT_2(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["k_GLUT"] = parvals["k_GLUT"]
    # p["tau_GLUT"] = parvals["tau_GLUT"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model_constant_modified, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    res = sol["pH"] - data
    return res


with open("input_SV.yml", "r") as f:
    p_init = yaml.full_load(f)

p = p_init["p"]
init = p_init["init"]

##################################################################################################
########################################## bare span #############################################
##################################################################################################

t_start = 0
t_stop = 2000
t_stop = 500
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
psi, psi_tot = models.calculate_psi_SV(sol, PP)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, sol["pH"])
ax[0].plot(t, data)
ax[0].set_ylim(5.7, 6.7)
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("pH_L []")
ax[0].legend(("simulation with fitted parameters", "fitting function"))
ax[1].plot(t, psi_tot * 1e3)
ax[1].set_ylim(-20, 40)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("psi [mV]")
ax[2].plot(t, sol["GABA"])
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("GABA molecules []")
plt.tight_layout()
plt.show()

##################################################################################################
########################################## GABA span #############################################
##################################################################################################

t_start = 0
t_stop = 2000
t_stop = 500
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
# params.add("tau_GABA_2", value=25, min=0)

fit_result = lmfit.minimize(
    residual_GABA, params, args=(t,), kws={"data": data, "p": P, "init": INIT}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P["k_GABA"] = parvals["k_GABA"]
P["tau_GABA"] = parvals["tau_GABA"]
# P["tau_GABA_2"] = parvals["tau_GABA_2"]

PP, y0 = models.set_SV_model(P, INIT)
y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, sol["pH"])
ax[0].plot(t, data)
ax[0].set_ylim(5.7, 6.7)
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("pH_L []")
ax[0].legend(("simulation with fitted parameters", "fitting function"))
ax[1].plot(t, psi_tot * 1e3)
ax[1].set_ylim(-20, 40)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("psi [mV]")
ax[2].plot(t, sol["GABA"])
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("GABA molecules []")
plt.tight_layout()
plt.show()

y = spi.odeint(models.SV_model_constant_modified, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, sol["pH"])
ax[0].plot(t, data)
ax[0].set_ylim(5.7, 6.7)
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("pH_L []")
# ax[0].legend(("simulation with fitted parameters", "fitting function"))
ax[1].plot(t, psi_tot * 1e3)
ax[1].set_ylim(-20, 40)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("psi [mV]")
ax[2].plot(t, sol["GABA"])
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("GABA molecules []")
plt.tight_layout()
plt.show()

params = lmfit.Parameters()
# params.add("k_GABA", value=50, min=0, max=75)
# params.add("tau_GABA", value=25, min=0)
params.add("tau_GABA_2", value=25, min=0)

fit_result = lmfit.minimize(
    residual_GABA_2, params, args=(t,), kws={"data": data, "p": P, "init": INIT}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
# P["k_GABA"] = parvals["k_GABA"]
# P["tau_GABA"] = parvals["tau_GABA"]
P["tau_GABA_2"] = parvals["tau_GABA_2"]

PP, y0 = models.set_SV_model(P, INIT)
y = spi.odeint(models.SV_model, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, sol["pH"])
ax[0].plot(t, data)
ax[0].set_ylim(5.7, 6.7)
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("pH_L []")
# ax[0].legend(("simulation with fitted parameters", "fitting function"))
ax[1].plot(t, psi_tot * 1e3)
ax[1].set_ylim(-20, 40)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("psi [mV]")
ax[2].plot(t, sol["GABA"])
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("GABA molecules []")
plt.tight_layout()
plt.show()

##################################################################################################
########################################## GLUT span #############################################
##################################################################################################

t_start = 0
t_stop = 500
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

data = GLUT_exp_decay(t)

INIT = init.copy()
INIT["pH_L"] = 6.6
P = p.copy()
P["N_VGAT"] = 0

params = lmfit.Parameters()
params.add("k_GLUT", value=7, min=0)
# params.add("tau_GLUT", value=1, min=0)

fit_result = lmfit.minimize(
    residual_GLUT, params, args=(t,), kws={"data": data, "p": P, "init": INIT}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P["k_GLUT"] = parvals["k_GLUT"]
# P["tau_GLUT"] = parvals["tau_GLUT"]

PP, y0 = models.set_SV_model(P, INIT)
y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, sol["pH"])
ax[0].plot(t, data)
ax[0].set_ylim(5.7, 6.7)
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("pH_L []")
ax[0].legend(("simulation with fitted parameters", "fitting function"))
ax[1].plot(t, psi_tot * 1e3)
ax[1].set_ylim(-20, 40)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("psi [mV]")
ax[2].plot(t, sol["GLUT"])
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("GLUT molecules []")
plt.tight_layout()
plt.show()

t_start = 0
t_stop = 2000
t_stop = 500
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

data = GLUT_exp_decay(t)

params = lmfit.Parameters()
params.add("k_GLUT", value=10, min=0)

fit_result = lmfit.minimize(
    residual_GLUT_2, params, args=(t,), kws={"data": data, "p": P, "init": INIT}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P["k_GLUT"] = parvals["k_GLUT"]

PP, y0 = models.set_SV_model(P, INIT)
y = spi.odeint(models.SV_model_constant_modified, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)

fig, ax = plt.subplots(3, 1)
ax[0].plot(t, sol["pH"])
ax[0].plot(t, data)
ax[0].set_ylim(5.7, 6.7)
ax[0].set_xlabel("time [s]")
ax[0].set_ylabel("pH_L []")
# ax[0].legend(("simulation with fitted parameters", "fitting function"))
ax[1].plot(t, psi_tot * 1e3)
ax[1].set_ylim(-20, 40)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("psi [mV]")
ax[2].plot(t, sol["GLUT"])
ax[2].set_xlabel("time [s]")
ax[2].set_ylabel("GLUT molecules []")
plt.tight_layout()
plt.show()
