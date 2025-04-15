import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import models
import lmfit


def fluo_exp_decay(x, tau, k):
    y = k * np.exp(-x / tau)
    return y


def residual_fluo(pars, t, data):
    parvals = pars.valuesdict()
    tau = parvals["tau"]
    k = parvals["k"]

    sol = fluo_exp_decay(t, tau, k)

    res = sol - data
    return res


def GLUT_exp_decay(x, tau):
    y = 5.8 + (6.6 - 5.8) * np.exp(-x / tau)
    return y


def residual_woClC(pars, t, data, p, init):
    parvals = pars.valuesdict()
    p["P_H"] = parvals["P_H"]
    p["Vmaxglu"] = parvals["Vmaxglu"]
    p["VmaxCl"] = parvals["VmaxCl"]

    p, y0 = models.set_SV_model(p, init)
    y = spi.odeint(models.SV_model_woClC, y0, t, args=(p,))
    sol = models.extract_solution_SV(y, p)
    psi, psi_tot = models.calculate_psi_SV(sol, p)

    res = sol["pH"] - data

    # print(np.min(psi_tot))
    # if np.any(psi_tot < -50e-3):
    # 	res += 1e3
    return res


def exp_filling(x, tau, k):
    y = k * (1 - np.exp(-x / tau))
    return y


def residual_filling(pars, t, data):
    parvals = pars.valuesdict()
    tau = parvals["tau"]
    k = parvals["k"]

    sol = exp_filling(t, tau, k)

    res = sol - data
    return res


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
t_stop = 2000
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

data = GLUT_exp_decay(t, tau_exp_GLUT)


path = "input_SV_woClC.yml"
p, init = models.read_input_file(path)

t_start = 0
t_stop = 2000
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

INIT = init.copy()
P = p.copy()
P["N_VGAT"] = 0
INIT["pH_L"] = 6.6

params = lmfit.Parameters()
params.add("P_H", value=P["P_H"], min=0, vary=False)
params.add("Vmaxglu", value=100, min=0)
params.add("VmaxCl", value=1000, min=0)
# params.add("Vmaxglu", value=P["Vmaxglu"], min=0)
# params.add("VmaxCl", value=P["VmaxCl"], min=0)

fit_result = lmfit.minimize(
    residual_woClC,
    params,
    args=(t,),
    kws={"data": data, "p": P.copy(), "init": INIT.copy()},
)
print(lmfit.fit_report(fit_result))
parvals = fit_result.params.valuesdict()
P_H = parvals["P_H"]
Vmaxglu = parvals["Vmaxglu"]
VmaxCl = parvals["VmaxCl"]


P["P_H"] = P_H
P["Vmaxglu"] = Vmaxglu
P["VmaxCl"] = VmaxCl

sol, psi, psi_tot = models.simulate_SV_model_woClC(P, INIT, t)
P, y0 = models.set_SV_model(P.copy(), INIT.copy())
psi, psi_tot, J_V = models.calculate_psi_SV2(sol, P)

fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(
    t, data, color="tab:blue", linestyle=":", label="Experimental pH", linewidth=2.5
)
ax1.plot(t, sol["pH"], color=color, label="pH")
ax1.set_ylim(5.6, 6.8)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)
# ax1.legend()

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi * 1e3, color=color, linestyle="--")
ax1.plot([], [], color=color, linestyle="--")
ax2.set_ylim(20, 120)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)
ax1.legend()

color = "red"
ax[1].plot(t, sol["GLUT"], color=color)
ax[1].set_ylim(0, 5000)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("GLUT molecules []")
plt.tight_layout()
plt.show()


params = lmfit.Parameters()
params.add("k", value=sol["GLUT"][-1], min=0)  # , vary=False)
params.add("tau", value=50, min=0)  # , vary=False)

fit_result = lmfit.minimize(
    residual_filling, params, args=(t,), kws={"data": sol["GLUT"]}
)
print(lmfit.fit_report(fit_result))
parvals = fit_result.params.valuesdict()
k = parvals["k"]
tau = parvals["tau"]

plt.plot(t, sol["GLUT"])
plt.plot(t, exp_filling(t, tau, k))
plt.show()


##################################################################################################

path = "input_SV_woClC.yml"
p, init = models.read_input_file(path)

t_start = 0
t_stop = 200
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

INIT = init.copy()
P = p.copy()
P["N_VGAT"] = 0

sol, psi, psi_tot = models.simulate_SV_model_woClC(P, INIT, t)
P, y0 = models.set_SV_model(P.copy(), INIT.copy())
psi, psi_tot, J_V = models.calculate_psi_SV2(sol, P)

fig, ax = plt.subplots(4, 1)

color = "red"
ax[0].plot(t, psi * 1e3, color=color)
ax[1].plot(t, sol["pH"], color=color)
ax[2].plot(t, sol["Cl"] * 1e3, color=color)
ax[3].plot(t, sol["GLUT"], color=color)


P["C_0"] = 1.45267584e-14 / P["S"]

sol, psi, psi_tot = models.simulate_SV_model_woClC(P, INIT, t)
P, y0 = models.set_SV_model(P.copy(), INIT.copy())
psi, psi_tot, J_V = models.calculate_psi_SV2(sol, P)

color = "black"
ax[0].plot(t, psi * 1e3, color=color)
ax[1].plot(t, sol["pH"], color=color)
ax[2].plot(t, sol["Cl"] * 1e3, color=color)
ax[3].plot(t, sol["GLUT"], color=color)

ax[0].set_xlabel("Time [s]")
ax[1].set_xlabel("Time [s]")
ax[2].set_xlabel("Time [s]")
ax[3].set_xlabel("Time [s]")

ax[0].set_ylabel("psi [mV]")
ax[1].set_ylabel("pH_L []")
ax[2].set_ylabel("Cl- [mM]")
ax[3].set_ylabel("GLUT molecules []")

ax[0].legend(["C_0 correct", "C_0 wrong"])
ax[1].legend(["C_0 correct", "C_0 wrong"])
ax[2].legend(["C_0 correct", "C_0 wrong"])
ax[3].legend(["C_0 correct", "C_0 wrong"])

plt.tight_layout()
plt.show()


##################################################################################################

path = "input_SV_woClC.yml"
p, init = models.read_input_file(path)

t_start = 0
t_stop = 200
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

INIT = init.copy()
P = p.copy()
P["N_VGAT"] = 0

sol, psi, psi_tot = models.simulate_SV_model_woClC(P, INIT, t)
P, y0 = models.set_SV_model(P.copy(), INIT.copy())
psi, psi_tot, J_V = models.calculate_psi_SV2(sol, P)

fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
# ax1.set_ylim(5, 7.5)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi * 1e3, color=color, linestyle="--")
# ax2.set_ylim(0, 80)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

color = "red"
ax[1].plot(t, sol["GLUT"], color=color)
ax[1].set_xlabel("time [s]")
ax[1].set_ylabel("GLUT molecules []")
plt.tight_layout()
plt.show()
