import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import models
import lmfit


def pH_exp_decay(x, tau):
    y = 5.98 + (6.6 - 5.98) * np.exp(-x / tau)
    return y


def fluo_exp_decay(x, tau, k):
    y = k * np.exp(-x / tau)
    return y


def residual_bare(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["P_H"] = parvals["P_H"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    res = sol["pH"][-1] - data[-1]
    return res


def residual_pH(pars, t, data):
    parvals = pars.valuesdict()
    tau = parvals["tau"]

    sol = pH_exp_decay(t, tau)

    res = sol - data
    return res


def residual_fluo(pars, t, data):
    parvals = pars.valuesdict()
    tau = parvals["tau"]
    k = parvals["k"]

    sol = fluo_exp_decay(t, tau, k)

    res = sol - data
    return res


def GABA_exp_decay(x, p, init):
    p["N_VGLUT"] = 0
    p["N_VGAT"] = 0
    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model, y0, t, args=(p,))
    sol_0 = models.extract_solution_SV(y, p)
    pH_0 = sol_0["pH"]

    tau = 25
    y = pH_0 + 0.42 * (1 - np.exp(-x / tau))
    return y


def GABA_exp_decay_2(x, tau):
    pH_0 = pH_exp_decay(x, tau)
    tau_alk = 25
    y = pH_0 + 0.42 * (1 - np.exp(-x / tau_alk))
    return y


def residual_GABA_K(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["k_GABA"] = parvals["k_GABA"]
    # p["tau_VGAT"] = parvals["tau_VGAT"]
    # p["tau_GABA"] = parvals["tau_GABA"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    res = sol["pH"] - data
    return res


def residual_GABA_K_tau(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["k_GABA"] = parvals["k_GABA"]
    p["tau_VGAT"] = parvals["tau_VGAT"]
    # p["tau_GABA"] = parvals["tau_GABA"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    res = sol["pH"] - data
    return res


def residual_GABA_tau2(pars, t, data, p, init):
    parvals = pars.valuesdict()
    # p["k_GABA"] = parvals["k_GABA"]
    # p["tau_VGAT"] = parvals["tau_VGAT"]
    p["tau_GABA"] = parvals["tau_GABA"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    molecules = 4000
    data = np.append(data, molecules)
    res = np.append(sol["pH"], sol["GABA"][-1]) - data
    # res = sol["pH"] - data
    return res


def residual_GABA_K_tau_tau2(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["k_GABA"] = parvals["k_GABA"]
    p["tau_VGAT"] = parvals["tau_VGAT"]
    p["tau_GABA"] = parvals["tau_GABA"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model_modified2, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    #molecules = 4000
    #data = np.append(data, molecules)
    #res = (np.append(sol["pH"], sol["GABA"][-1]) - data) / data
    #res = sol["pH"] - data
    res = sol["GABA"][-10:] - 4000
    return res


def GLUT_exp_decay(x, tau):
    y = 5.8 + (6.6 - 5.8) * np.exp(-x / tau)
    return y


def residual_GLUT(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["k_GLUT"] = parvals["k_GLUT"]
    # p["tau_VGLUT"] = parvals["tau_VGLUT"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    res = sol["pH"] - data
    return res


def residual_GLUT_2(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["k_GLUT"] = parvals["k_GLUT"]
    p["tau_VGLUT"] = parvals["tau_VGLUT"]
    p["P_Cl_VGLUT"] = parvals["P_Cl_VGLUT"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)

    res = sol["pH"] - data
    # res = np.append(sol["pH"] - data, (sol["Cl"][-1] - 0.015)/0.015)
    return res


with open("input_SV.yml", "r") as f:
    p_init = yaml.full_load(f)

p = p_init["p"]
init = p_init["init"]

##################################################################################################
########################################## bare span #############################################
##################################################################################################

exp_pH_bareSV = pd.read_csv("experimental_pH_bareSV.csv", header=None)
exp_pH_bareSV = np.array(exp_pH_bareSV)
t_exp = exp_pH_bareSV[:, 0]
t_exp = t_exp - t_exp[0]
pH_exp = exp_pH_bareSV[:, 1]

params = lmfit.Parameters()
params.add("tau", value=0.1, min=0)

fit_result = lmfit.minimize(residual_pH, params, args=(t_exp,), kws={"data": pH_exp})
print(lmfit.fit_report(fit_result))
parvals = fit_result.params.valuesdict()
tau_exp_bareSV = parvals["tau"]

plt.plot(t_exp, pH_exp)
plt.plot(t_exp, pH_exp_decay(t_exp, tau_exp_bareSV))
plt.show()


t_start = 0
t_stop = 2000
t_stop = 500
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

data = pH_exp_decay(t, tau=tau_exp_bareSV)
# plt.plot(t, data)
# plt.show()

INIT = init.copy()
INIT["pH_L"] = 6.6
P = p.copy()
P["N_VGLUT"] = 0
P["N_VGAT"] = 0
P["tau_VGAT"] = 0
P["tau_GABA"] = 0
P["tau_VGLUT"] = 0

params = lmfit.Parameters()
params.add("P_H", value=0.1, min=0)

fit_result = lmfit.minimize(
    residual_bare, params, args=(t,), kws={"data": data, "p": P.copy(), "init": INIT.copy()}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P_H_ref = parvals["P_H"]
P["P_H"] = P_H_ref

PP, y0 = models.set_SV_model(P.copy(), INIT.copy())
y = spi.odeint(models.SV_model, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)


fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.plot(
    t, data, color="tab:blue", linestyle=":", label="Experimental pH", linewidth=2.5
)
# ax1.set_xlim(0.5, 160)
ax1.set_ylim(5.6, 7)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 40)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

color = "green"
ax[1].plot(t, sol["GABA"], color=color)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("GABA molecules []")
plt.tight_layout()
plt.show()


params = lmfit.Parameters()
params.add("tau", value=0.1, min=0)

fit_result = lmfit.minimize(residual_pH, params, args=(t,), kws={"data": sol["pH"]})
print(lmfit.fit_report(fit_result))
parvals = fit_result.params.valuesdict()
# tau = parvals["tau"]

# plt.plot(t, sol["pH"])
# plt.plot(t, pH_exp_decay(t, tau))

# plt.show()

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
P["tau_VGAT"] = 0
P["tau_GABA"] = 0
P["tau_VGLUT"] = 0

data = GABA_exp_decay_2(t, tau_exp_bareSV)
# plt.plot(t, data)
# plt.show()

INIT = init.copy()
INIT["pH_L"] = 6.6
P = p.copy()
P["P_H"] = P_H_ref
P["tau_VGAT"] = 0
P["tau_GABA"] = 0
P["tau_VGLUT"] = 0
P["N_VGLUT"] = 0

params = lmfit.Parameters()
params.add("k_GABA", value=50, min=0)

fit_result = lmfit.minimize(
    residual_GABA_K, params, args=(t,), kws={"data": data, "p": P.copy(), "init": INIT.copy()}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P["k_GABA"] = parvals["k_GABA"]

PP, y0 = models.set_SV_model(P.copy(), INIT.copy())
y = spi.odeint(models.SV_model, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)


fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.plot(
    t, data, color="tab:blue", linestyle=":", label="Experimental pH", linewidth=2.5
)
# ax1.set_xlim(0.5, 160)
ax1.set_ylim(5.6, 7)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.legend()

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 40)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

color = "green"
ax[1].plot(t, sol["GABA"], color=color)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("GABA molecules []")
plt.tight_layout()
plt.show()


params = lmfit.Parameters()
params.add("k_GABA", value=50, min=0)
params.add("tau_VGAT", value=25, min=0)

fit_result = lmfit.minimize(
    residual_GABA_K_tau, params, args=(t,), kws={"data": data, "p": P.copy(), "init": INIT.copy()}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P["k_GABA"] = parvals["k_GABA"]
P["tau_VGAT"] = parvals["tau_VGAT"]

PP, y0 = models.set_SV_model(P.copy(), INIT.copy())
y = spi.odeint(models.SV_model, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)


fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.plot(
    t, data, color="tab:blue", linestyle=":", label="Experimental pH", linewidth=2.5
)
# ax1.set_xlim(0.5, 160)
ax1.set_ylim(5.6, 7)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.legend()

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 40)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

color = "green"
ax[1].plot(t, sol["GABA"], color=color)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("GABA molecules []")
plt.tight_layout()
plt.show()


y = spi.odeint(models.SV_model_modified, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)


fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.plot(
    t, data, color="tab:blue", linestyle=":", label="Experimental pH", linewidth=2.5
)
# ax1.set_xlim(0.5, 160)
ax1.set_ylim(5.6, 7)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.legend()

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 40)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

color = "green"
ax[1].plot(t, sol["GABA"], color=color)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("GABA molecules []")
plt.tight_layout()
plt.show()


params = lmfit.Parameters()
params.add("tau_GABA", value=25, min=0)

fit_result = lmfit.minimize(
    residual_GABA_tau2, params, args=(t,), kws={"data": data, "p": P.copy(), "init": INIT.copy()}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P["tau_GABA"] = parvals["tau_GABA"]

PP, y0 = models.set_SV_model(P.copy(), INIT.copy())
y = spi.odeint(models.SV_model, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)


fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.plot(
    t, data, color="tab:blue", linestyle=":", label="Experimental pH", linewidth=2.5
)
# ax1.set_xlim(0.5, 160)
ax1.set_ylim(5.6, 7)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.legend()

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 40)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

color = "green"
ax[1].plot(t, sol["GABA"], color=color)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("GABA molecules []")
plt.tight_layout()
plt.show()


params = lmfit.Parameters()
params.add("k_GABA", value=50, min=0)
params.add("tau_VGAT", value=25, min=0)
params.add("tau_GABA", value=100, min=0)
#params.add("k_GABA", value=P["k_GABA"], min=0)
#params.add("tau_VGAT", value=P["tau_VGAT"], min=0)
#params.add("tau_GABA", value=4000/P["k_GABA"], min=0)

fit_result = lmfit.minimize(
    residual_GABA_K_tau_tau2, params, args=(t,), kws={"data": data, "p": P.copy(), "init": INIT.copy()}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P["k_GABA"] = parvals["k_GABA"]
P["tau_VGAT"] = parvals["tau_VGAT"]
P["tau_GABA"] = parvals["tau_GABA"]

#P["tau_GABA"] = 4000/P["k_GABA"]

PP, y0 = models.set_SV_model(P.copy(), INIT.copy())
y = spi.odeint(models.SV_model_modified2, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)


fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.plot(
    t, data, color="tab:blue", linestyle=":", label="Experimental pH", linewidth=2.5
)
# ax1.set_xlim(0.5, 160)
ax1.set_ylim(5.6, 7)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.legend()

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 40)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

color = "green"
ax[1].plot(t, sol["GABA"], color=color)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("GABA molecules []")
plt.tight_layout()
plt.show()

##################################################################################################
########################################## GLUT span #############################################
##################################################################################################

exp_fluo_GLUT = pd.read_csv("experimental_fluo_GLUT.csv", header=None)
exp_fluo_GLUT = np.array(exp_fluo_GLUT)
t_exp = exp_fluo_GLUT[:, 0]
t_exp = t_exp - t_exp[0]
fluo_exp = exp_fluo_GLUT[:, 1]

params = lmfit.Parameters()
params.add("tau", value=10, min=0)
params.add("k", value=5, min=0)

fit_result = lmfit.minimize(
    residual_fluo, params, args=(t_exp,), kws={"data": fluo_exp}
)
print(lmfit.fit_report(fit_result))
parvals = fit_result.params.valuesdict()
tau_exp_GLUT = parvals["tau"]
k_exp = parvals["k"]

plt.plot(t_exp, fluo_exp)
plt.plot(t_exp, fluo_exp_decay(t_exp, tau_exp_GLUT, k_exp))
plt.show()


t_start = 0
t_stop = 500
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

data = GLUT_exp_decay(t, tau_exp_GLUT)

INIT = init.copy()
INIT["pH_L"] = 6.6
P = p.copy()
P["P_H"] = P_H_ref
P["tau_VGAT"] = 0
P["tau_GABA"] = 0
P["tau_VGLUT"] = 0
P["N_VGAT"] = 0

params = lmfit.Parameters()
params.add("k_GLUT", value=1, min=0)

fit_result = lmfit.minimize(
    residual_GLUT, params, args=(t,), kws={"data": data, "p": P.copy(), "init": INIT.copy()}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
P["k_GLUT"] = parvals["k_GLUT"]

PP, y0 = models.set_SV_model(P.copy(), INIT.copy())
y = spi.odeint(models.SV_model, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)


fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.plot(
    t, data, color="tab:blue", linestyle=":", label="Experimental pH", linewidth=2.5
)
# ax1.set_xlim(0.5, 160)
ax1.set_ylim(5.6, 7)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.legend()

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 40)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

color = "red"
ax[1].plot(t, sol["GLUT"], color=color)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("GLUT molecules []")
plt.tight_layout()
plt.show()


t_start = 0
t_stop = 500
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

data = GLUT_exp_decay(t, tau_exp_GLUT)

INIT = init.copy()
INIT["pH_L"] = 6.6
P = p.copy()
P["P_H"] = P_H_ref
P["tau_VGAT"] = 0
P["tau_GABA"] = 0
P["tau_VGLUT"] = 0
P["N_VGAT"] = 0

params = lmfit.Parameters()
params.add("k_GLUT", value=1, min=0, max=150)
params.add("tau_VGLUT", value=150, min=0, max=200)
params.add("P_Cl_VGLUT", value=0, vary=False)

fit_result = lmfit.minimize(
    residual_GLUT_2, params, args=(t,), kws={"data": data, "p": P.copy(), "init": INIT.copy()}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
k_GLUT = parvals["k_GLUT"]
tau_VGLUT = parvals["tau_VGLUT"]
P_Cl_VGLUT = parvals["P_Cl_VGLUT"]
P["k_GLUT"] = k_GLUT
P["tau_VGLUT"] = tau_VGLUT
P["P_Cl_VGLUT"] = P_Cl_VGLUT

PP, y0 = models.set_SV_model(P.copy(), INIT.copy())
y = spi.odeint(models.SV_model, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)


fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.plot(
    t, data, color="tab:blue", linestyle=":", label="Experimental pH", linewidth=2.5
)
# ax1.set_xlim(0.5, 160)
ax1.set_ylim(5.6, 7)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.legend()

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 40)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

color = "red"
ax[1].plot(t, sol["GLUT"], color=color)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("GLUT molecules []")
plt.tight_layout()
plt.show()

plt.plot(t, sol["Cl"])
plt.show()


t_start = 0
t_stop = 500
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

data = GLUT_exp_decay(t, tau_exp_GLUT)

INIT = init.copy()
INIT["pH_L"] = 6.6
P = p.copy()
P["P_H"] = P_H_ref
P["tau_VGAT"] = 0
P["tau_GABA"] = 0
P["tau_VGLUT"] = 0
P["N_VGAT"] = 0

params = lmfit.Parameters()
params.add("k_GLUT", value=10, min=0, max=150)
params.add("tau_VGLUT", value=100, min=0, max=150)
params.add("P_Cl_VGLUT", value=1e-8, min=1e-9, max=1e-6)

fit_result = lmfit.minimize(
    residual_GLUT_2, params, args=(t,), kws={"data": data, "p": P.copy(), "init": INIT.copy()}
)
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
k_GLUT = parvals["k_GLUT"]
tau_VGLUT = parvals["tau_VGLUT"]
P_Cl_VGLUT = parvals["P_Cl_VGLUT"]
P["k_GLUT"] = k_GLUT
P["tau_VGLUT"] = tau_VGLUT
P["P_Cl_VGLUT"] = P_Cl_VGLUT

PP, y0 = models.set_SV_model(P.copy(), INIT.copy())
y = spi.odeint(models.SV_model, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, psi_tot = models.calculate_psi_SV(sol, PP)


fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.plot(
    t, data, color="tab:blue", linestyle=":", label="Experimental pH", linewidth=2.5
)
# ax1.set_xlim(0.5, 160)
ax1.set_ylim(5.6, 7)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.legend()

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 40)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

color = "red"
ax[1].plot(t, sol["GLUT"], color=color)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("GLUT molecules []")
plt.tight_layout()
plt.show()

plt.plot(t, sol["Cl"])
plt.show()



