import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import yaml
import pandas as pd
import models
from scipy.interpolate import RegularGridInterpolator
import lmfit

def GLUT_exp_decay(x):
	tau = 25
	y = 5.8 + (7.4 - 5.8) * np.exp (- x/tau)
	return y

def residual_GLUT(pars, t, data):
	with open("input_SV.yml", "r") as f:
		p_init = yaml.full_load(f)

	p = p_init["p"]
	p["N_VGAT"] = 0
	init = p_init["init"]
	
	parvals = pars.valuesdict()
	p["k_GLUT"] = parvals["k_GLUT"]
	p["tau_GLUT"] = parvals["tau_GLUT"]
	
	p, y0 = models.set_SV_model(p, init)
	y = spi.odeint(models.SV_model_constant, y0, t, args=(p,))
	sol = models.extract_solution_SV(y, p)
	
	res = sol["pH"] - data
	print(p["k_GLUT"], p["tau_GLUT"])
	#plt.plot(t, sol["pH"])
	#plt.plot(t, data)
	#plt.show()
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

    y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))

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

y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))

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

t_start = 0
t_stop = 2000
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

data = GLUT_exp_decay(t)

params = lmfit.Parameters()
params.add('k_GLUT', value=7, min=0)
params.add('tau_GLUT', value=1)

fit_result = lmfit.minimize(residual_GLUT, params, args=(t,), kws={'data': data})
print(lmfit.fit_report(fit_result))

INIT = init.copy()
P = p.copy()
P["N_VGAT"] = 0

parvals = fit_result.params.valuesdict()
P["k_GLUT"] = parvals["k_GLUT"]
P["tau_GLUT"] = parvals["tau_GLUT"]

PP, y0 = models.set_SV_model(P, INIT)
y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))
sol = models.extract_solution_SV(y, PP)
psi, _ = models.calculate_psi_SV(sol, PP)

plt.plot(t, sol['pH'])
plt.plot(t, data)
plt.xlabel("time [s]")
plt.show()

fig, ax = plt.subplots(3,1)
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


##################################################################################################
########################################## GLUT span #############################################
##################################################################################################

t_start = 0
t_stop = 2000
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

fig, ax = plt.subplots(2,1, subplot_kw={"projection": "3d"})

K = [i for i in range(10, 210, 10)]
#K = [i * 1e-1 for i in range(10, 151)]
#K = [i * 1e-1 for i in range(10, 81)]
TAU = [i for i in range(1, 21, 1)]

K, TAU = np.meshgrid(K, TAU)

P = p.copy()
P["N_VGAT"] = 0

INIT = init.copy()

pH_SS = np.zeros(K.shape)
GLUT_SS = np.zeros(K.shape)
tau_acid = np.zeros(K.shape)

for i in range(K.shape[0]):
	for j in range(K.shape[1]):
		k = K[i,j]
		tau = TAU[i,j]
		
		P["k_GLUT"] = k
		P["tau_GLUT"] = tau

		PP, y0 = models.set_SV_model(P, INIT)
		
		try:
			y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))

			sol = models.extract_solution_SV(y, PP)

			pH_SS[i,j] = sol["pH"][-1]
			GLUT_SS[i,j] = sol["GLUT"][-1]
		except:
			pass

K = np.array(K)
pH_SS = np.array(pH_SS)
pH_ref = 5.8
#ind = np.nonzero(pH_SS > pH_ref)[0][-1]
#K_ref = K[ind] + ( (K[ind+1]-K[ind])/(pH_SS[ind+1]-pH_SS[ind]) ) * (pH_ref-pH_SS[ind])
#print(K_ref)

ax[0].plot_surface(K, TAU, pH_SS)
#ax.plot([6e-5, P_H_ref, P_H_ref], [pH_ref, pH_ref, 4.5], color='k', linestyle='--')
#ax.set_xlim(6e-5, 6e-3)
#ax.set_ylim(4.5, 7.5)
ax[0].set_xlabel("GLUT transport rate [s^-1]")
ax[0].set_ylabel("GLUT efflux time constant [s]")
ax[0].set_zlabel("Final pH_L []")

ax[1].plot_surface(K, TAU, GLUT_SS)
#ax.plot([6e-5, P_H_ref, P_H_ref], [pH_ref, pH_ref, 4.5], color='k', linestyle='--')
#ax.set_xlim(6e-5, 6e-3)
#ax.set_ylim(4.5, 7.5)
ax[1].set_xlabel("GLUT transport rate [s^-1]")
ax[1].set_ylabel("GLUT efflux time constant [s]")
ax[1].set_zlabel("Final GLUT molecules []")
plt.tight_layout()
plt.show()

##################################################################################################
########################################## GLUT #############################################
##################################################################################################

t_start = 0
t_stop = 2000
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

fig, ax = plt.subplots(3,1)

P = p.copy()
P["N_VGAT"] = 0
P["k_GLUT"] = 150
P["tau_GLUT"] = 15

INIT = init.copy()

PP, y0 = models.set_SV_model(P, INIT)

y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))

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


##################################################################################################
########################################## GLUT span #############################################
##################################################################################################

t_start = 0
t_stop = 700
dt = 0.02
t = np.arange(start=t_start, stop=t_stop, step=dt)

fig, ax = plt.subplots(2,1)

K = [i * 1e-1 for i in range(10, 151)]
#K = [i * 1e-1 for i in range(10, 81)]

P = p.copy()
P["N_VGAT"] = 0

INIT = init.copy()

pH_SS = list()
GLUT_SS = list()

for k in K:
    P["k_GLUT"] = k

    PP, y0 = models.set_SV_model(P, INIT)

    y = spi.odeint(models.SV_model_constant, y0, t, args=(PP,))

    sol = models.extract_solution_SV(y, PP)

    pH_SS.append(sol["pH"][-1])
    GLUT_SS.append(sol["GLUT"][-1])

K = np.array(K)
pH_SS = np.array(pH_SS)
pH_ref = 5.8
ind = np.nonzero(pH_SS > pH_ref)[0][-1]
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




