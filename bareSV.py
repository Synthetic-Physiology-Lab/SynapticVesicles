import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import models
from scipy.interpolate import RegularGridInterpolator
import lmfit


def pH_exp_decay(x, tau):
    y = 5.98 + (6.6 - 5.98) * np.exp(-x / tau)
    return y


def residual_pH(pars, t, data):
    parvals = pars.valuesdict()
    tau = parvals["tau"]

    sol = pH_exp_decay(t, tau)

    res = sol - data
    return res


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
INIT["pH_L"] = 6.6

pH_SS = list()

for h_p in H_P:
    P["P_H"] = h_p

    PP, y0 = models.set_bareSV_model(P.copy(), INIT.copy())

    y = spi.odeint(models.bareSV_model, y0, t, args=(PP,))

    sol = models.extract_solution_bareSV(y, PP)

    pH_SS.append(sol["pH"][-1])

H_P = np.array(H_P)
pH_SS = np.array(pH_SS)
pH_ref = 5.98
ind = np.nonzero(pH_SS < pH_ref)[0][-1]
P_H_ref = H_P[ind] + ((H_P[ind + 1] - H_P[ind]) / (pH_SS[ind + 1] - pH_SS[ind])) * (
    pH_ref - pH_SS[ind]
)
print(P_H_ref)

color = "gray"
ax.plot(H_P, pH_SS, color=color)
ax.plot([6e-5, P_H_ref, P_H_ref], [pH_ref, pH_ref, 4.5], color="k", linestyle=":")
ax.set_xlim(6e-5, 6e-3)
ax.set_ylim(4.5, 7.5)
ax.set_xlabel("H+ permeability [cm/s]")
ax.set_ylabel("Final pH_L []")
plt.show()

##################################################################################################
######################################## time constant ###########################################
##################################################################################################

exp_pH_bareSV = pd.read_csv("experimental_pH_bareSV.csv", header=None)
exp_pH_bareSV = np.array(exp_pH_bareSV)
# print(exp_pH_bareSV)
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


P = p.copy()
P["P_H"] = P_H_ref

INIT = init.copy()
INIT["pH_L"] = 6.6

PP, y0 = models.set_bareSV_model(P.copy(), INIT.copy())

y = spi.odeint(models.bareSV_model, y0, t, args=(PP,))

sol = models.extract_solution_bareSV(y, PP)
data = sol["pH"]

params = lmfit.Parameters()
params.add("tau", value=10, min=0)

fit_result = lmfit.minimize(residual_pH, params, args=(t,), kws={"data": data})
print(lmfit.fit_report(fit_result))

parvals = fit_result.params.valuesdict()
tau = parvals["tau"]

fit = pH_exp_decay(t, tau)

plt.plot(t, sol["pH"])
plt.plot(t, fit)
plt.show()

##################################################################################################
########################################## FIGURE 2e #############################################
##################################################################################################

t_start = 0
t_stop = 100
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)


data = pH_exp_decay(t, tau=tau_exp_bareSV)


P = p.copy()
P["P_H"] = P_H_ref

INIT = init.copy()
INIT["pH_L"] = 6.6

PP, y0 = models.set_bareSV_model(P.copy(), INIT.copy())

y = spi.odeint(models.bareSV_model, y0, t, args=(PP,))

sol = models.extract_solution_bareSV(y, PP)

psi, psi_tot = models.calculate_psi_bareSV(sol, PP)

fig, ax1 = plt.subplots()

color = "gray"
ax1.plot(t, sol["pH"], color=color)
ax1.plot(
    t, data, color="tab:blue", linestyle=":", label="Experimental pH", linewidth=2.5
)
# ax1.set_xlim(0, 100)
# ax1.set_ylim(5.5, 7.5)
ax1.set_xlabel("time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)
ax1.legend()

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
# ax2.set_ylim(0, 80)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()
plt.show()

##################################################################################################
########################################## FIGURE 2f #############################################
##################################################################################################

beta = [i * 1e-1 for i in range(4, 1601, 2)]

P = p.copy()
P["P_H"] = P_H_ref

INIT = init.copy()
INIT["pH_L"] = 6.6

pH_SS = list()
psi_SS = list()

for b in beta:
    P["beta"] = b * 1e-3

    PP, y0 = models.set_bareSV_model(P.copy(), INIT.copy())

    y = spi.odeint(models.bareSV_model, y0, t, args=(PP,))

    sol = models.extract_solution_bareSV(y, PP)

    psi, psi_tot = models.calculate_psi_bareSV(sol, PP)

    pH_SS.append(sol["pH"][-1])
    psi_SS.append(psi_tot[-1] * 1e3)

fig, ax1 = plt.subplots()

beta = np.array(beta)
pH_SS = np.array(pH_SS)
psi_SS = np.array(psi_SS)

RMSE_pH = np.sqrt(
    np.mean((pH_SS[(beta >= 20) & (beta <= 60)] - pH_SS[beta == 40]) ** 2)
)
RMSE_psi = np.sqrt(
    np.mean((psi_SS[(beta >= 20) & (beta <= 60)] - psi_SS[beta == 40]) ** 2)
)

RMSE_pH_perc = RMSE_pH / pH_SS[beta == 40] * 100
RMSE_psi_perc = RMSE_psi / psi_SS[beta == 40] * 100

# diff_pH_perc = np.abs(pH_SS[beta == 20] - pH_SS[beta == 60]) / pH_SS[beta == 40] * 100
# diff_psi_perc = (
#    np.abs(psi_SS[beta == 20] - psi_SS[beta == 60]) / psi_SS[beta == 40] * 100
# )

# print(RMSE_pH, RMSE_psi)
print(RMSE_pH_perc, RMSE_psi_perc)
# print(diff_pH_perc, diff_psi_perc)

color = "gray"
ax1.plot(beta, pH_SS, color=color)
ax1.plot([20, 20, 60, 60, 20], [5.9, 6.05, 6.05, 5.9, 5.9], color="k", linestyle=":")
# ax1.set_xlim(0.5, 160)
# ax1.set_ylim(5.9, 6.5)
ax1.set_xlabel("Buffering capacity [mM/pH]")
ax1.set_ylabel("Final pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)

color = "black"
ax2 = ax1.twinx()
ax2.plot(beta, psi_SS, color=color, linestyle="--")
# ax2.set_ylim(35, 65)
ax2.set_ylabel("Final psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

fig.tight_layout()
plt.show()

##################################################################################################
########################################## FIGURE 3a #############################################
##################################################################################################

P = p.copy()
P["P_H"] = P_H_ref

INIT = init.copy()
INIT["pH_L"] = 6.6

PP, y0 = models.set_bareSV_model(P.copy(), INIT.copy())

y = spi.odeint(models.bareSV_model, y0, t, args=(PP,))

sol = models.extract_solution_bareSV(y, PP)

psi, psi_tot = models.calculate_psi_bareSV(sol, PP)

H_flow = models.calculate_Hflow_bareSV(sol, PP)

fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.set_ylim(5.5, 8)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 100)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

colors = ["orange", "purple", "black"]
for i in range(len(H_flow.keys())):
    key = list(H_flow.keys())[i]
    color = colors[i]
    ax[1].plot(t, H_flow[key], label=key, color=color)
# ax[1].legend()
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("H+ flow [H+/s]")
ax[1].set_ylim(-150, 160)

fig.tight_layout()
plt.show()

##################################################################################################
########################################## FIGURE 3b #############################################
##################################################################################################

P = p.copy()
P["P_H"] = P_H_ref
P["N_ClC"] = 0

INIT = init.copy()
INIT["pH_L"] = 6.6

PP, y0 = models.set_bareSV_model(P.copy(), INIT.copy())

y = spi.odeint(models.bareSV_model, y0, t[t <= 100], args=(PP,))

sol = models.extract_solution_bareSV(y, PP)

psi, psi_tot = models.calculate_psi_bareSV(sol, PP)

H_flow = models.calculate_Hflow_bareSV(sol, PP)


fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.set_ylim(5.5, 8)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 100)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

colors = ["orange", "purple", "black"]
for i in range(len(H_flow.keys())):
    key = list(H_flow.keys())[i]
    color = colors[i]
    ax[1].plot(t, H_flow[key], label=key, color=color)
# ax[1].legend()
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("H+ flow [H+/s]")
ax[1].set_ylim(-150, 160)

fig.tight_layout()
plt.show()

##################################################################################################
########################################## FIGURE 3c #############################################
##################################################################################################

P = p.copy()
P["P_H"] = P_H_ref
P["N_V"] = 0

INIT = init.copy()
INIT["pH_L"] = 6.6

PP, y0 = models.set_bareSV_model(P.copy(), INIT.copy())

y = spi.odeint(models.bareSV_model, y0, t[t <= 100], args=(PP,))

sol = models.extract_solution_bareSV(y, PP)

psi, psi_tot = models.calculate_psi_bareSV(sol, PP)

H_flow = models.calculate_Hflow_bareSV(sol, PP)


fig, ax = plt.subplots(2, 1)

color = "gray"
ax1 = ax[0]
ax1.plot(t, sol["pH"], color=color)
ax1.set_ylim(5.5, 8)
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("pH_L []", color=color)
ax1.tick_params(axis="y", labelcolor=color)

color = "black"
ax2 = ax1.twinx()
ax2.plot(t, psi_tot * 1e3, color=color, linestyle="--")
ax2.set_ylim(-10, 100)
ax2.set_ylabel("psi [mV]", color=color)
ax2.tick_params(axis="y", labelcolor=color)

colors = ["orange", "purple", "black"]
for i in range(len(H_flow.keys())):
    key = list(H_flow.keys())[i]
    color = colors[i]
    ax[1].plot(t, H_flow[key], label=key, color=color)
ax[1].legend()
ax[1].set_xlabel("Time [s]")
ax[1].set_ylabel("H+ flow [H+/s]")
ax[1].set_ylim(-150, 160)

fig.tight_layout()
plt.show()
