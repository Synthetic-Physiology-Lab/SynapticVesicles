import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def lysosome_model(y, t, p):

    ## parameters extraction
    # Physics constants
    k_b = p["k_b"]  # Boltzmann constant
    R = p["R"]  # Gas constant [J / (mol * K)]
    F = p["F"]  # Faraday's constant [C / mol]
    N_A = p["N_A"]  # Avogadro constant [mol^-1]

    # Temperature
    T = p["T"] + 273.15  # Absolute temperature [K]

    # Cytosolic concentrations
    pH_C = p["pH_C"]  # Cytosolic pH []
    H_C = 10 ** (-pH_C)  # Cytosolic H+ concentration [M]
    K_C = p["K_C"]  # Cytosolic K+ concentration [M]
    Na_C = p["Na_C"]  # Cytosolic Na+ concentration [M]
    Cl_C = p["Cl_C"]  # Cytosolic Cl- concentration [M] 5-50

    # Permeabilities
    P_H = p["P_H"]  # H+ permeability [cm/s]
    P_K = p["P_K"]  # K+ permeability [cm/s]
    P_Na = p["P_Na"]  # Na+ permeability [cm/s]
    P_Cl = p["P_Cl"]  # Cl- permeability [cm/s]
    P_W = p["P_W"]  # H2O permeability [cm/s]

    # Capacitance density
    C_0 = p["C_0"]  # Lipid bilayer capacitance [F/cm^2]

    # Leaflets potentials
    psi_o = p["psi_o"]  # Outside leaflet potential [V]
    psi_i = p["psi_i"]  # Inside leaflet potential [V]

    # Lysosome dimensions
    d = p["d"]  # Lysosome diameter [um]
    S = p["S"]  # Lysosome surface area [cm^2]
    V_0 = p["V_0"]  # Lysosome inital volume [L]

    # Luminal buffering capacity
    beta = p["beta"]  # Buffering capacity [M/pH]

    # Luminal concentration of impermeant charges
    B = p["B"]  # Concentration of impermeant charges [M]

    # Osmotic parameter
    theta = p["theta"]  # Osmotic coefficient []
    v_W = p["v_W"]  # Partial molar volume of water [cm^3/mol]
    theta_C = p["theta_C"]  # Cytoplasmic osmolyte concentration [M]

    # Pumps quantity
    N_V = p["N_V"]  # Number of V-ATPases
    N_ClC = p["N_ClC"]  # Number of ClC-7 antiporters
    J_VATP = p["J_VATP"]  # V-ATPase flux [nterpolator object

    # ClC-7 pump stoichiometry
    ClC_Cl = p["ClC_Cl"]  # ClC-7 Cl- Stoichiometry
    ClC_H = p["ClC_H"]  # ClC-7 H+ Stoichiometry

    ## state variables extraction
    H, pH, K, Na, Cl, V = y  # Ionic species are expressed in number of molecules
    # they are converted in concentrations [M]
    H = H / V / N_A
    K = K / V / N_A
    Na = Na / V / N_A
    Cl = Cl / V / N_A

    # Modified cytoplasmic surface concentrations
    # Cle = Cl_C * np.exp(psi_o / RTF)
    # Ke = K_C * np.exp(-psi_o / RTF)
    # Nae = Na_C * np.exp(-psi_o / RTF)
    # pHe = pHbulk + psi_o / (RTF * 2.3) # 2.3 = ln(10)

    # Modified luminal surface concentrations
    # Cli = Cl * np.exp(psi_i / RTF)
    # Ki = K * np.exp(-psi_i / RTF)
    # Nai = Na * np.exp(-psi_i / RTF)
    # pHi = pH + psi_i / (RTF * 2.3) # 2.3 = ln(10)

    ## parts calculation
    psi = F / (C_0 * S) * (V * (K + Na - Cl + beta * (pH_C - pH)) - B * V_0)
    # U = psi / (k_b * T)
    RTF = R * T / F
    U = psi / RTF
    a = 0.3
    b = 1.5e-5
    print("############# ", Cl_C, Cl, " ############")
    delta_u_ClC = (ClC_Cl + 1) * psi + RTF * (
        2.3 * (pH - pH_C) + ClC_Cl * np.log(Cl_C / Cl)
    )
    x = 0.5 * (1 + np.tanh((delta_u_ClC + 250) / 75))
    J_ClC = x * a * delta_u_ClC + (1 - x) * b * delta_u_ClC**3
    J_V = J_VATP([psi, pH])[0]

    J_H = P_H * S * (U * (H - (H_C * np.exp(-U)))) / (1 - np.exp(-U))
    J_K = P_K * S * (U * (K - (K_C * np.exp(-U)))) / (1 - np.exp(-U))
    J_Na = P_Na * S * (U * (Na - (Na_C * np.exp(-U)))) / (1 - np.exp(-U))
    J_Cl = P_Cl * S * (U * (Cl - (Cl_C * np.exp(-U)))) / (1 - np.exp(-U))

    ## derivatives calculation
    dH = N_V * J_V + N_ClC * J_ClC + J_H
    dpH = 1 / beta * dH / V / N_A
    dK = +J_K
    dNa = +J_Na
    dCl = -2 * N_ClC * J_ClC + J_Cl
    dV = P_W * S * v_W * (theta * (H + K + Na + Cl) - theta_C)

    dy = (dH, dpH, dK, dNa, dCl, dV)
    print("############# ", dy, " ############")
    return np.array(dy)


def lysosome_model_MADONNA(y: np.ndarray, t: np.ndarray, p: dict):
    """Lysosome model corresponding to Berkeley Madonna code.


    Parameters
    ----------

    y: np.ndarray
        Initial conditions
    t: np.ndarray
        Time points
    p: dict
        Model parameters

    Notes
    -----

    Model parameters are checked for completeness in function TODO.
    """

    ## parameters extraction
    # Physics constants
    k_b = p["k_b"]  # Boltzmann constant
    R = p["R"]  # Gas constant [J / (mol * K)]
    F = p["F"]  # Faraday's constant [C / mol]
    N_A = p["N_A"]  # Avogadro constant [mol^-1]

    # Temperature
    T = p["T"] + 273.15  # Absolute temperature [K]

    # Cytosolic concentrations
    pH_C = p["pH_C"]  # Cytosolic pH []
    H_C = 10 ** (-pH_C)  # Cytosolic H+ concentration [M]
    K_C = p["K_C"]  # Cytosolic K+ concentration [M]
    Na_C = p["Na_C"]  # Cytosolic Na+ concentration [M]
    Cl_C = p["Cl_C"]  # Cytosolic Cl- concentration [M] 5-50

    # Permeabilities
    P_H = p["P_H"]  # H+ permeability [cm/s]
    P_K = p["P_K"]  # K+ permeability [cm/s]
    P_Na = p["P_Na"]  # Na+ permeability [cm/s]
    P_Cl = p["P_Cl"]  # Cl- permeability [cm/s]
    P_W = p["P_W"]  # H2O permeability [cm/s]

    # Capacitance density
    C_0 = p["C_0"]  # Lipid bilayer capacitance [F/cm^2]

    # Leaflets potentials
    psi_o = p["psi_o"]  # Outside leaflet potential [V]
    psi_i = p["psi_i"]  # Inside leaflet potential [V]

    # Lysosome dimensions
    d = p["d"]  # Lysosome diameter [um]
    S = p["S"]  # Lysosome surface area [cm^2]
    V_0 = p["V_0"] * 1e-15  # Lysosome inital volume [um^3]

    # Luminal buffering capacity
    beta = p["beta"]  # Buffering capacity [M/pH]

    # Luminal concentration of impermeant charges
    B = p["B"]  # Concentration of impermeant charges [M]

    # Osmotic balance term
    Q = p["Q"]

    # Osmotic parameter
    theta = p["theta"]  # Osmotic coefficient []
    v_W = p["v_W"]  # Partial molar volume of water [cm^3/mol]
    theta_C = p["theta_C"]  # Cytoplasmic osmolyte concentration [M]

    # Pumps quantity
    N_V = p["N_V"]  # Number of V-ATPases
    N_ClC = p["N_ClC"]  # Number of ClC-7 antiporters
    J_VATP = p["J_VATP"]  # V-ATPase flux [nterpolator object

    # ClC-7 pump stoichiometry
    ClC_Cl = p["ClC_Cl"]  # ClC-7 Cl- Stoichiometry
    ClC_H = p["ClC_H"]  # ClC-7 H+ Stoichiometry

    ## state variables extraction
    V, pH, H, K, Na, Cl = y  # Ionic species are expressed in number of molecules
    # they are converted in concentrations [M]
    V = V * 1e-15
    H = H / V / N_A
    K = K / V / N_A
    Na = Na / V / N_A
    Cl = Cl / V / N_A

    RTF = R * T / F
    # Modified cytoplasmic surface concentrations
    Cle = Cl_C * np.exp(psi_o / RTF)
    Ke = K_C * np.exp(-psi_o / RTF)
    Nae = Na_C * np.exp(-psi_o / RTF)
    pHe = pH_C + psi_o / (RTF * 2.3)  # 2.3 = ln(10)

    # Modified luminal surface concentrations
    Cli = Cl * np.exp(psi_i / RTF)
    Ki = K * np.exp(-psi_i / RTF)
    Nai = Na * np.exp(-psi_i / RTF)
    pHi = pH + psi_i / (RTF * 2.3)  # 2.3 = ln(10)

    ## parts calculation
    psi = F / (C_0 * S) * (V * (H + K + Na - Cl) - B * V_0)
    U = psi / RTF
    a = -0.3
    b = -1.5e-5
    delta_u_ClC = (ClC_Cl + ClC_H) * psi * 1e3 + RTF * 1e3 * (
        2.3 * (pHe - pHi) + ClC_Cl * np.log(Cle / Cli)
    )
    x = 0.5 + 0.5 * np.tanh((delta_u_ClC + 250) / 75)
    J_ClC = x * a * delta_u_ClC + (1 - x) * b * delta_u_ClC**3
    J_V = J_VATP([psi * 1e3, pH])[0]

    if np.abs(psi) > 0.01:
        gg = U / (1 - np.exp(-U))
    else:
        gg = 1 / (1 - U / 2 + U**2 / 6 - U**3 / 24 + U**4 / 120)

    J_H = P_H * S * gg * (10 ** (-pHe) * np.exp(-U) - 10 ** (-pHi)) * N_A / 1000
    J_K = P_K * S * gg * (Ke * np.exp(-U) - Ki) * N_A / 1000
    J_Na = P_Na * S * gg * (Nae * np.exp(-U) - Nai) * N_A / 1000
    J_Cl = P_Cl * S * gg * (Cle - Cli * np.exp(-U)) * N_A / 1000
    J_W = P_W * S * (theta * (10 ** (-pH) + K + Na + Cl) + Q / V - theta_C)

    ## derivatives calculation
    dV = J_W * v_W / 1e6 * 1e15  # / 1000 / 55 = * v_W(18) / 1e6
    dpH = -(N_V * J_V + N_ClC * J_ClC + J_H) / beta / V / N_A
    dH = N_V * J_V + ClC_H * N_ClC * J_ClC + J_H
    dK = J_K
    dNa = J_Na
    dCl = J_Cl - ClC_Cl * N_ClC * J_ClC

    dy = (dV, dpH, dH, dK, dNa, dCl)
    return np.array(dy)


def set_lysosome_model_MADONNA(p, init):
    if p["d"] > 0:
        r = p["d"] * 1e-6 / 2  # Lysosome radius [m]
        V = 4 / 3 * np.pi * r**3 * 1e18  # SV volume [um^3]
        # V = 4 / 3 * np.pi * r**3 * 1e3  # Lysosome volume [L]
        S = 4 * np.pi * r**2 * 1e4  # Lysosome surface area [cm^2]
    else:
        V = p["V"] * 1e15  # Lysosome volume [um^3]
        # V = p["V"]  # Lysosome volume [L]
        S = p["S"]  # Lysosome surface area [cm^2]

    B = (
        init["K_L"]
        + init["Na_L"]
        - init["Cl_L"]
        + init["H_L"]
        + (p["C_0"] * S) / (V * 1e-15 * p["F"]) * (p["psi_o"] - p["psi_i"])
    )

    Q = (
        V
        * 1e-15
        * (
            p["theta_C"]
            - p["theta"]
            * (10 ** (-init["pH_L"]) + init["K_L"] + init["Na_L"] + init["Cl_L"])
        )
    )

    p["S"] = S
    p["B"] = B
    p["Q"] = Q
    init["V"] = V
    p["V_0"] = V

    PP = pd.read_excel("datasetProtonPump.xlsx", header=None)
    PP = np.array(PP)
    psi = PP[1:, 0]
    pH = PP[0, 1:]
    values = PP[1:, 1:]
    PP_interp = RegularGridInterpolator(
        points=(psi, pH), values=values, method="linear"
    )
    p["J_VATP"] = PP_interp

    # Ionic species are expressed in concentrations [M]
    # they are converted in number of molecules
    N_A = p["N_A"]
    y0 = (
        init["V"],
        init["pH_L"],
        init["H_L"] * V * 1e-15 * N_A,
        init["K_L"] * V * 1e-15 * N_A,
        init["Na_L"] * V * 1e-15 * N_A,
        init["Cl_L"] * V * 1e-15 * N_A,
    )
    y0 = np.array(y0)

    return (p, y0)


def extract_solution(y, p):
    N_A = p["N_A"]
    V = y[:, 0]
    V_liters = V * 1e-15
    pH = y[:, 1]
    H = y[:, 2] / V_liters / N_A
    K = y[:, 3] / V_liters / N_A
    Na = y[:, 4] / V_liters / N_A
    Cl = y[:, 5] / V_liters / N_A

    sol = dict()
    sol["V"] = V
    sol["pH"] = pH
    sol["H"] = H
    sol["K"] = K
    sol["Na"] = Na
    sol["Cl"] = Cl

    return sol


def calculate_psi(sol, p):
    psi = (
        p["F"]
        / (p["C_0"] * p["S"])
        * (sol["V"] * (sol["H"] + sol["K"] + sol["Na"] - sol["Cl"]) - p["B"] * p["V_0"])
    )
    psi_tot = psi + p["psi_o"] - p["psi_i"]
    return (psi, psi_tot)


def bareSV_model(y: np.ndarray, t: np.ndarray, p: dict):
    """bare synaptic vesicle model containing just vATPase, ClC-3 and passive proton leakage.


    Parameters
    ----------

    y: np.ndarray
        Initial conditions
    t: np.ndarray
        Time points
    p: dict
        Model parameters

    Notes
    -----

    Model parameters are checked for completeness in function TODO.
    """

    ## parameters extraction
    # Physics constants
    k_b = p["k_b"]  # Boltzmann constant
    R = p["R"]  # Gas constant [J / (mol * K)]
    F = p["F"]  # Faraday's constant [C / mol]
    N_A = p["N_A"]  # Avogadro constant [mol^-1]

    # Temperature
    T = p["T"] + 273.15  # Absolute temperature [K]

    # Cytosolic concentrations
    pH_C = p["pH_C"]  # Cytosolic pH []
    H_C = 10 ** (-pH_C)  # Cytosolic H+ concentration [M]
    K_C = p["K_C"]  # Cytosolic K+ concentration [M]
    Na_C = p["Na_C"]  # Cytosolic Na+ concentration [M]
    Cl_C = p["Cl_C"]  # Cytosolic Cl- concentration [M] 5-50

    # Luminal concentrations
    K = p["K_L"]  # Luminal K+ concentration [M]
    Na = p["Na_L"]  # Luminal Na+ concentration [M]

    # Permeabilities
    P_H = p["P_H"]  # H+ permeability [cm/s]
    # P_Cl = p["P_Cl"]  # Cl- permeability [cm/s]
    P_W = p["P_W"]  # H2O permeability [cm/s]

    # Capacitance density
    C_0 = p["C_0"]  # Lipid bilayer capacitance [F/cm^2]

    # Leaflets potentials
    psi_o = p["psi_o"]  # Outside leaflet potential [V]
    psi_i = p["psi_i"]  # Inside leaflet potential [V]

    # Lysosome dimensions
    d = p["d"]  # Lysosome diameter [um]
    S = p["S"]  # Lysosome surface area [cm^2]
    V_0 = p["V_0"] * 1e-15  # Lysosome initial volume [um^3 to L]

    # Luminal buffering capacity
    beta = p["beta"]  # Buffering capacity [M/pH]

    # Luminal concentration of impermeant charges
    B = p["B"]  # Concentration of impermeant charges [M]

    # Osmotic balance term
    Q = p["Q"]

    # Osmotic parameter
    theta = p["theta"]  # Osmotic coefficient []
    v_W = p["v_W"]  # Partial molar volume of water [cm^3/mol]
    theta_C = p["theta_C"]  # Cytoplasmic osmolyte concentration [M]

    # Pumps quantity
    N_V = p["N_V"]  # Number of V-ATPases
    N_ClC = p["N_ClC"]  # Number of ClC-7 antiporters
    J_VATP = p["J_VATP"]  # V-ATPase flux [H+/s] (interpolator object)

    # ClC-7 pump stoichiometry
    ClC_Cl = p["ClC_Cl"]  # ClC-7 Cl- Stoichiometry
    ClC_H = p["ClC_H"]  # ClC-7 H+ Stoichiometry

    ## state variables extraction
    V, pH, H, Cl = y  # Ionic species are expressed in number of molecules
    # they are converted in concentrations [M]
    V = V * 1e-15 # [um^3 to L]
    H = H / V / N_A
    Cl = Cl / V / N_A

    RTF = R * T / F
    # Modified cytoplasmic surface concentrations
    Cle = Cl_C * np.exp(psi_o / RTF)
    pHe = pH_C + psi_o / (RTF * 2.3)  # 2.3 = ln(10)

    # Modified luminal surface concentrations
    Cli = Cl * np.exp(psi_i / RTF)
    pHi = pH + psi_i / (RTF * 2.3)  # 2.3 = ln(10)

    ## parts calculation
    psi = F / (C_0 * S) * (V * (H + K + Na - Cl) - B * V_0)
    U = psi / RTF
    a = -0.3
    b = -1.5e-5
    delta_u_ClC = (ClC_Cl + ClC_H) * psi * 1e3 + RTF * 1e3 * (
        2.3 * (pHe - pHi) + ClC_Cl * np.log(Cle / Cli)
    )
    x = 0.5 + 0.5 * np.tanh((delta_u_ClC + 250) / 75)
    J_ClC = x * a * delta_u_ClC + (1 - x) * b * delta_u_ClC**3
    J_V = J_VATP([psi * 1e3, pH])[0]

    if np.abs(psi) > 0.01e-3:
        gg = U / (1 - np.exp(-U))
    else:
        gg = 1 / (1 - U / 2 + U**2 / 6 - U**3 / 24 + U**4 / 120)

    J_H = P_H * S * gg * (10 ** (-pHe) * np.exp(-U) - 10 ** (-pHi)) * N_A / 1000
    # J_Cl = P_Cl * S * gg * (Cle - Cli * np.exp(-U)) * N_A / 1000
    J_W = P_W * S * (theta * (10 ** (-pH) + K + Na + Cl) + Q / V - theta_C)

    ## derivatives calculation
    dV = J_W * v_W / 1e6 * 1e15 # [L/s to um^3/s]
    dpH = -(N_V * J_V + ClC_H * N_ClC * J_ClC + J_H) / beta / V / N_A
    dH = N_V * J_V + ClC_H * N_ClC * J_ClC + J_H
    dCl = -ClC_Cl * N_ClC * J_ClC  # + J_Cl

    dy = (dV, dpH, dH, dCl)
    return np.array(dy)


def set_bareSV_model(p, init):
    if p["d"] > 0:
        r = p["d"] * 1e-6 / 2  # SV radius [m]
        V = 4 / 3 * np.pi * r**3 * 1e18  # SV volume [um^3]
        # V = 4 / 3 * np.pi * r**3 * 1e3  # SV volume [L]
        S = 4 * np.pi * r**2 * 1e4  # SV surface area [cm^2]
    else:
        V = p["V"] * 1e15  # Lysosome volume [um^3]
        # V = p["V"]  # Lysosome volume [L]
        S = p["S"]  # Lysosome surface area [cm^2]

    B = (
        p["K_L"]
        + p["Na_L"]
        - init["Cl_L"]
        + init["H_L"]
        + (p["C_0"] * S) / (V * 1e-15 * p["F"]) * (p["psi_o"] - p["psi_i"])
    )

    Q = (
        V
        * 1e-15
        * (
            p["theta_C"]
            - p["theta"] * (10 ** (-init["pH_L"]) + p["K_L"] + p["Na_L"] + init["Cl_L"])
        )
    )

    p["S"] = S
    p["B"] = B
    p["Q"] = Q
    init["V"] = V
    p["V_0"] = V

    PP = pd.read_excel("datasetProtonPump.xlsx", header=None)
    PP = np.array(PP)
    psi = PP[1:, 0]
    pH = PP[0, 1:]
    values = PP[1:, 1:]
    PP_interp = RegularGridInterpolator(
        points=(psi, pH), values=values, method="linear"
    )
    p["J_VATP"] = PP_interp

    # Ionic species are expressed in concentrations [M]
    # they are converted in number of molecules
    N_A = p["N_A"]
    y0 = (
        init["V"],
        init["pH_L"],
        init["H_L"] * V * 1e-15 * N_A,
        init["Cl_L"] * V * 1e-15 * N_A,
    )
    y0 = np.array(y0)

    return (p, y0)


def extract_solution_bareSV(y, p):
    N_A = p["N_A"]
    V = y[:, 0]
    V_liters = V * 1e-15
    pH = y[:, 1]
    H = y[:, 2] / V_liters / N_A
    Cl = y[:, 3] / V_liters / N_A

    sol = dict()
    sol["V"] = V
    sol["pH"] = pH
    sol["H"] = H
    sol["Cl"] = Cl

    return sol



def calculate_psi_bareSV(sol, p):
	## parameters extraction
    F = p["F"]  # Faraday's constant [C / mol]
    N_A = p["N_A"]  # Avogadro constant [mol^-1]

    K = p["K_L"]  # Luminal K+ concentration [M]
    Na = p["Na_L"]  # Luminal Na+ concentration [M]

    C_0 = p["C_0"]  # Lipid bilayer capacitance [F/cm^2]

    psi_o = p["psi_o"]  # Outside leaflet potential [V]
    psi_i = p["psi_i"]  # Inside leaflet potential [V]

    S = p["S"]  # Lysosome surface area [cm^2]
    V_0 = p["V_0"] * 1e-15  # Lysosome initial volume [um^3 to L]

    B = p["B"]  # Concentration of impermeant charges [M]

    V = sol["V"] * 1e-15 # [um^3 to L]
    H = sol["H"]
    Cl = sol["Cl"]
    
    psi = F / (C_0 * S) * (V * (H + K + Na - Cl) - B * V_0)
    psi_tot = psi + psi_o - psi_i

    return (psi, psi_tot)



def calculate_Hflow_bareSV(sol, p):
	#psi, _ = calculate_psi_bareSV(sol, p)
	
	## parameters extraction
    # Physics constants
    k_b = p["k_b"]  # Boltzmann constant
    R = p["R"]  # Gas constant [J / (mol * K)]
    F = p["F"]  # Faraday's constant [C / mol]
    N_A = p["N_A"]  # Avogadro constant [mol^-1]

    # Temperature
    T = p["T"] + 273.15  # Absolute temperature [K]

    # Cytosolic concentrations
    pH_C = p["pH_C"]  # Cytosolic pH []
    H_C = 10 ** (-pH_C)  # Cytosolic H+ concentration [M]
    Cl_C = p["Cl_C"]  # Cytosolic Cl- concentration [M] 5-50

    # Luminal concentrations
    K = p["K_L"]  # Luminal K+ concentration [M]
    Na = p["Na_L"]  # Luminal Na+ concentration [M]

    # Permeabilities
    P_H = p["P_H"]  # H+ permeability [cm/s]

    # Capacitance density
    C_0 = p["C_0"]  # Lipid bilayer capacitance [F/cm^2]

    # Leaflets potentials
    psi_o = p["psi_o"]  # Outside leaflet potential [V]
    psi_i = p["psi_i"]  # Inside leaflet potential [V]

    # Lysosome dimensions
    d = p["d"]  # Lysosome diameter [um]
    S = p["S"]  # Lysosome surface area [cm^2]
    V_0 = p["V_0"] * 1e-15  # Lysosome initial volume [um^3 to L]

    # Luminal buffering capacity
    beta = p["beta"]  # Buffering capacity [M/pH]

    # Luminal concentration of impermeant charges
    B = p["B"]  # Concentration of impermeant charges [M]

    # Pumps quantity
    N_V = p["N_V"]  # Number of V-ATPases
    N_ClC = p["N_ClC"]  # Number of ClC-7 antiporters
    J_VATP = p["J_VATP"]  # V-ATPase flux [H+/s] (interpolator object)

    # ClC-7 pump stoichiometry
    ClC_Cl = p["ClC_Cl"]  # ClC-7 Cl- Stoichiometry
    ClC_H = p["ClC_H"]  # ClC-7 H+ Stoichiometry

    # Ionic species are expressed in number of molecules
    # they are converted in concentrations [M]
    V = sol["V"] * 1e-15 # [um^3 to L]
    pH = sol["pH"]
    H = sol["H"]
    Cl = sol["Cl"]

    RTF = R * T / F
    # Modified cytoplasmic surface concentrations
    Cle = Cl_C * np.exp(psi_o / RTF)
    pHe = pH_C + psi_o / (RTF * 2.3)  # 2.3 = ln(10)

    # Modified luminal surface concentrations
    Cli = Cl * np.exp(psi_i / RTF)
    pHi = pH + psi_i / (RTF * 2.3)  # 2.3 = ln(10)

    ## parts calculation
    psi = F / (C_0 * S) * (V * (H + K + Na - Cl) - B * V_0)
    U = psi / RTF
    a = -0.3
    b = -1.5e-5
    delta_u_ClC = (ClC_Cl + ClC_H) * psi * 1e3 + RTF * 1e3 * (
        2.3 * (pHe - pHi) + ClC_Cl * np.log(Cle / Cli)
    )
    x = 0.5 + 0.5 * np.tanh((delta_u_ClC + 250) / 75)
    J_ClC = x * a * delta_u_ClC + (1 - x) * b * delta_u_ClC**3
    J_V = J_VATP(np.transpose(np.array([psi * 1e3, pH])))

    gg = list()
    for i in range(len(psi)):
        if np.abs(psi[i]) > 0.01e-3:
            gg.append( U[i] / (1 - np.exp(-U[i])) )
        else:
            gg.append( 1 / (1 - U[i] / 2 + U[i]**2 / 6 - U[i]**3 / 24 + U[i]**4 / 120) )
    gg = np.array(gg)

    J_H = P_H * S * gg * (10 ** (-pHe) * np.exp(-U) - 10 ** (-pHi)) * N_A / 1000
    
    #dH = N_V * J_V + ClC_H * N_ClC * J_ClC + J_H
    
    Hflow = dict()
    Hflow['vATPase'] = N_V * J_V
    Hflow['ClC3'] = ClC_H * N_ClC * J_ClC
    Hflow['leak'] = J_H
	
    return Hflow



def SV_model(y: np.ndarray, t: np.ndarray, p: dict):
    """synaptic vesicle model containing vATPase, ClC-3 and passive proton leakage, VGAT and VGLUT-1


    Parameters
    ----------

    y: np.ndarray
        Initial conditions
    t: np.ndarray
        Time points
    p: dict
        Model parameters

    Notes
    -----

    Model parameters are checked for completeness in function TODO.
    """

    ## parameters extraction
    # Physics constants
    k_b = p["k_b"]  # Boltzmann constant
    R = p["R"]  # Gas constant [J / (mol * K)]
    F = p["F"]  # Faraday's constant [C / mol]
    N_A = p["N_A"]  # Avogadro constant [mol^-1]

    # Temperature
    T = p["T"] + 273.15  # Absolute temperature [K]

    # Cytosolic concentrations
    pH_C = p["pH_C"]  # Cytosolic pH []
    H_C = 10 ** (-pH_C)  # Cytosolic H+ concentration [M]
    K_C = p["K_C"]  # Cytosolic K+ concentration [M]
    Na_C = p["Na_C"]  # Cytosolic Na+ concentration [M]
    Cl_C = p["Cl_C"]  # Cytosolic Cl- concentration [M] 5-50

    # Luminal concentrations
    K = p["K_L"]  # Luminal K+ concentration [M]
    Na = p["Na_L"]  # Luminal Na+ concentration [M]

    # Permeabilities
    P_H = p["P_H"]  # H+ permeability [cm/s]
    P_Cl = p["P_Cl"]  # Cl- permeability [cm/s]
    P_W = p["P_W"]  # H2O permeability [cm/s]

    # Capacitance density
    C_0 = p["C_0"]  # Lipid bilayer capacitance [F/cm^2]

    # Leaflets potentials
    psi_o = p["psi_o"]  # Outside leaflet potential [V]
    psi_i = p["psi_i"]  # Inside leaflet potential [V]

    # Lysosome dimensions
    d = p["d"]  # Lysosome diameter [um]
    S = p["S"]  # Lysosome surface area [cm^2]
    V_0 = p["V_0"] * 1e-15  # Lysosome initial volume [um^3 to L]

    # Luminal buffering capacity
    beta = p["beta"]  # Buffering capacity [M/pH]

    # Luminal concentration of impermeant charges
    B = p["B"]  # Concentration of impermeant charges [M]

    # Osmotic balance term
    Q = p["Q"]

    # Osmotic parameter
    theta = p["theta"]  # Osmotic coefficient []
    v_W = p["v_W"]  # Partial molar volume of water [cm^3/mol]
    theta_C = p["theta_C"]  # Cytoplasmic osmolyte concentration [M]

    # Pumps quantity
    N_V = p["N_V"]  # Number of V-ATPases
    N_ClC = p["N_ClC"]  # Number of ClC-7 antiporters
    J_VATP = p["J_VATP"]  # V-ATPase flux [H+/s] (interpolator object)
    N_VGAT = p["N_VGAT"] # Number of GABA transporters VGAT
    N_VGLUT = p["N_VGLUT"] # Number of glutamate transporters VGLUT-1

    # ClC-7 pump stoichiometry
    ClC_Cl = p["ClC_Cl"]  # ClC-7 Cl- Stoichiometry
    ClC_H = p["ClC_H"]  # ClC-7 H+ Stoichiometry
    
    # VGAT transporter stoichiometry
    VGAT_GABA = p["VGAT_GABA"] # VGAT GABA Stoichiometry
    VGAT_H = p["VGAT_H"] # VGAT H+ Stoichiometry

    # VGLUT-1 transporter stoichiometry
    VGLUT_GLUT = p["VGLUT_GLUT"] # VGLUT gluatamate Stoichiometry
    VGLUT_H = p["VGLUT_H"] # VGLUT H+ Stoichiometry

    # Neurotransmitters trasnsport rates
    k_GABA = p["k_GABA"] # GABA transport rate [s^-1]
    k_GLUT = p["k_GLUT"] # Glutamate transport rate [s^-1]

    ## state variables extraction
    V, pH, H, Cl, GABA, GLUT = y  # Ionic species are expressed in number of molecules
    # they are converted in concentrations [M]
    #print(GABA)
    V = V * 1e-15 # [um^3 to L]
    H = H / V / N_A
    Cl = Cl / V / N_A
    GABA = GABA / V / N_A
    GLUT = GLUT / V / N_A

    RTF = R * T / F
    # Modified cytoplasmic surface concentrations
    Cle = Cl_C * np.exp(psi_o / RTF)
    pHe = pH_C + psi_o / (RTF * 2.3)  # 2.3 = ln(10)

    # Modified luminal surface concentrations
    Cli = Cl * np.exp(psi_i / RTF)
    pHi = pH + psi_i / (RTF * 2.3)  # 2.3 = ln(10)

    ## parts calculation
    psi = F / (C_0 * S) * (V * (H + K + Na - Cl - GLUT) - B * V_0)
    #print('psi = ', psi)
    U = psi / RTF
    a = -0.3
    b = -1.5e-5
    delta_u_ClC = (ClC_Cl + ClC_H) * psi * 1e3 + RTF * 1e3 * (
        2.3 * (pHe - pHi) + ClC_Cl * np.log(Cle / Cli)
    )
    x = 0.5 + 0.5 * np.tanh((delta_u_ClC + 250) / 75)
    J_ClC = x * a * delta_u_ClC + (1 - x) * b * delta_u_ClC**3
    J_V = J_VATP([psi * 1e3, pH])[0]

    if np.abs(psi) > 0.01e-3:
        gg = U / (1 - np.exp(-U))
    else:
        gg = 1 / (1 - U / 2 + U**2 / 6 - U**3 / 24 + U**4 / 120)

    J_H = P_H * S * gg * (10 ** (-pHe) * np.exp(-U) - 10 ** (-pHi)) * N_A / 1000
    J_Cl = P_Cl * S * gg * (Cle - Cli * np.exp(-U)) * N_A / 1000
    J_W = P_W * S * (theta * (10 ** (-pH) + K + Na + Cl) + Q / V - theta_C)
    
    #delta_u_H = 6.4 - pH if pH < 6.4 else 0
    #delta_u_H = 2.3 * RTF * 1e3 * (pHe - pH) # + psi * 1e3
    #J_GABA = k_GABA*10 * delta_u_H
    #print(delta_u_H)
    ### option 1 (works)
    #J_GABA = N_VGAT * k_GABA * (10 ** (-pH)) * 1.2e8
    #dGABA = VGAT_GABA * J_GABA - GABA * V * N_A / 100
    ###
    ### option 2
    #V_max = 75
    #K_M = (10 ** (-6.0))/1
    #J_GABA = N_VGAT * k_GABA * V_max * (10 ** (-pH)) / (K_M + (10 ** (-pH)))
    #dGABA = VGAT_GABA * J_GABA - GABA * V * N_A / 100
    ###
    T = 60
    J_GABA = k_GABA * (1-np.exp(-t/35)) #* (t/T) if t < T else k_GABA #* 1 / (1 + np.exp(-10*(6.8-pH))) # * (1 + 10*(6.4 - pH))
    #print(J_GABA)
    J_GLUT = k_GLUT

    ## derivatives calculation
    H_tot = N_V * J_V + ClC_H * N_ClC * J_ClC + J_H - VGAT_H * N_VGAT * J_GABA - VGLUT_H * N_VGLUT * J_GLUT
    dV = J_W * v_W / 1e6 * 1e15 # [L/s to um^3/s]
    dpH = - H_tot / beta / V / N_A
    dH = H_tot
    dCl = -ClC_Cl * N_ClC * J_ClC - VGLUT_GLUT * N_VGLUT * J_GLUT #+ J_Cl
    #dGABA = VGAT_GABA * J_GABA - GABA * V * N_A / 100
    dGABA = VGAT_GABA * N_VGAT * J_GABA #- GABA * V * N_A * k_GABA / 5000 #* 1/25 #tau=25
    dGLUT = VGLUT_GLUT * N_VGLUT * J_GLUT - GLUT * V * N_A * k_GLUT * N_VGLUT / 1800
    #print(dGABA)

    dy = (dV, dpH, dH, dCl, dGABA, dGLUT)
    return np.array(dy)



def SV_model_constant(y: np.ndarray, t: np.ndarray, p: dict):
    """synaptic vesicle model containing vATPase, ClC-3 and passive proton leakage, VGAT and VGLUT-1


    Parameters
    ----------

    y: np.ndarray
        Initial conditions
    t: np.ndarray
        Time points
    p: dict
        Model parameters

    Notes
    -----

    Model parameters are checked for completeness in function TODO.
    """

    ## parameters extraction
    # Physics constants
    k_b = p["k_b"]  # Boltzmann constant
    R = p["R"]  # Gas constant [J / (mol * K)]
    F = p["F"]  # Faraday's constant [C / mol]
    N_A = p["N_A"]  # Avogadro constant [mol^-1]

    # Temperature
    T = p["T"] + 273.15  # Absolute temperature [K]

    # Cytosolic concentrations
    pH_C = p["pH_C"]  # Cytosolic pH []
    H_C = 10 ** (-pH_C)  # Cytosolic H+ concentration [M]
    K_C = p["K_C"]  # Cytosolic K+ concentration [M]
    Na_C = p["Na_C"]  # Cytosolic Na+ concentration [M]
    Cl_C = p["Cl_C"]  # Cytosolic Cl- concentration [M] 5-50

    # Luminal concentrations
    K = p["K_L"]  # Luminal K+ concentration [M]
    Na = p["Na_L"]  # Luminal Na+ concentration [M]

    # Permeabilities
    P_H = p["P_H"]  # H+ permeability [cm/s]
    # P_Cl = p["P_Cl"]  # Cl- permeability [cm/s]
    P_W = p["P_W"]  # H2O permeability [cm/s]

    # Capacitance density
    C_0 = p["C_0"]  # Lipid bilayer capacitance [F/cm^2]

    # Leaflets potentials
    psi_o = p["psi_o"]  # Outside leaflet potential [V]
    psi_i = p["psi_i"]  # Inside leaflet potential [V]

    # Lysosome dimensions
    d = p["d"]  # Lysosome diameter [um]
    S = p["S"]  # Lysosome surface area [cm^2]
    V_0 = p["V_0"] * 1e-15  # Lysosome initial volume [um^3 to L]

    # Luminal buffering capacity
    beta = p["beta"]  # Buffering capacity [M/pH]

    # Luminal concentration of impermeant charges
    B = p["B"]  # Concentration of impermeant charges [M]

    # Osmotic balance term
    Q = p["Q"]

    # Osmotic parameter
    theta = p["theta"]  # Osmotic coefficient []
    v_W = p["v_W"]  # Partial molar volume of water [cm^3/mol]
    theta_C = p["theta_C"]  # Cytoplasmic osmolyte concentration [M]

    # Pumps quantity
    N_V = p["N_V"]  # Number of V-ATPases
    N_ClC = p["N_ClC"]  # Number of ClC-7 antiporters
    J_VATP = p["J_VATP"]  # V-ATPase flux [H+/s] (interpolator object)
    N_VGAT = p["N_VGAT"] # Number of GABA transporters VGAT
    N_VGLUT = p["N_VGLUT"] # Number of glutamate transporters VGLUT-1

    # ClC-7 pump stoichiometry
    ClC_Cl = p["ClC_Cl"]  # ClC-7 Cl- Stoichiometry
    ClC_H = p["ClC_H"]  # ClC-7 H+ Stoichiometry
    
    # VGAT transporter stoichiometry
    VGAT_GABA = p["VGAT_GABA"] # VGAT GABA Stoichiometry
    VGAT_H = p["VGAT_H"] # VGAT H+ Stoichiometry

    # VGLUT-1 transporter stoichiometry
    VGLUT_GLUT = p["VGLUT_GLUT"] # VGLUT gluatamate Stoichiometry
    VGLUT_H = p["VGLUT_H"] # VGLUT H+ Stoichiometry

    # Neurotransmitters trasnsport rates
    k_GABA = p["k_GABA"] # GABA transport rate [s^-1]
    k_GLUT = p["k_GLUT"] # Glutamate transport rate [s^-1]
    tau_GLUT = p["tau_GLUT"] # Glutamate efflux time constant [s]

    ## state variables extraction
    V, pH, H, Cl, GABA, GLUT = y  # Ionic species are expressed in number of molecules
    # they are converted in concentrations [M]
    #print(GABA)
    V = V * 1e-15 # [um^3 to L]
    H = H / V / N_A
    Cl = Cl / V / N_A
    GABA = GABA / V / N_A
    GLUT = GLUT / V / N_A

    RTF = R * T / F
    # Modified cytoplasmic surface concentrations
    Cle = Cl_C * np.exp(psi_o / RTF)
    pHe = pH_C + psi_o / (RTF * 2.3)  # 2.3 = ln(10)

    # Modified luminal surface concentrations
    Cli = Cl * np.exp(psi_i / RTF)
    pHi = pH + psi_i / (RTF * 2.3)  # 2.3 = ln(10)

    ## parts calculation
    psi = F / (C_0 * S) * (V * (H + K + Na - Cl - GLUT) - B * V_0)
    #print('psi = ', psi)
    U = psi / RTF
    a = -0.3
    b = -1.5e-5
    delta_u_ClC = (ClC_Cl + ClC_H) * psi * 1e3 + RTF * 1e3 * (
        2.3 * (pHe - pHi) + ClC_Cl * np.log(Cle / Cli)
    )
    x = 0.5 + 0.5 * np.tanh((delta_u_ClC + 250) / 75)
    J_ClC = x * a * delta_u_ClC + (1 - x) * b * delta_u_ClC**3
    J_V = J_VATP([psi * 1e3, pH])[0]

    if np.abs(psi) > 0.01e-3:
        gg = U / (1 - np.exp(-U))
    else:
        gg = 1 / (1 - U / 2 + U**2 / 6 - U**3 / 24 + U**4 / 120)

    J_H = P_H * S * gg * (10 ** (-pHe) * np.exp(-U) - 10 ** (-pHi)) * N_A / 1000
    # J_Cl = P_Cl * S * gg * (Cle - Cli * np.exp(-U)) * N_A / 1000
    J_W = P_W * S * (theta * (10 ** (-pH) + K + Na + Cl) + Q / V - theta_C)
    
    J_GABA = k_GABA #* (pH-6.4) #(1 / (1 + np.exp(-10*(pH-6.4))))
    J_GLUT = k_GLUT #* (pH-5.8) #(1 / (1 + np.exp(-10*(pH-5.8))))

    ## derivatives calculation
    H_tot = N_V * J_V + ClC_H * N_ClC * J_ClC + J_H - VGAT_H * N_VGAT * J_GABA #- VGLUT_H * N_VGLUT * J_GLUT
    dV = J_W * v_W / 1e6 * 1e15 # [L/s to um^3/s]
    dpH = - H_tot / beta / V / N_A
    dH = H_tot
    dCl = -ClC_Cl * N_ClC * J_ClC - VGLUT_GLUT * N_VGLUT * J_GLUT # + J_Cl
    dGABA = VGAT_GABA * N_VGAT * J_GABA
    dGLUT = VGLUT_GLUT * N_VGLUT * J_GLUT - GLUT * V * N_A / tau_GLUT#* k_GLUT * N_VGLUT / 1800
    #print(dGABA)

    dy = (dV, dpH, dH, dCl, dGABA, dGLUT)
    return np.array(dy)



def set_SV_model(p, init):
    if p["d"] > 0:
        r = p["d"] * 1e-6 / 2  # SV radius [m]
        V = 4 / 3 * np.pi * r**3 * 1e18  # SV volume [um^3]
        # V = 4 / 3 * np.pi * r**3 * 1e3  # SV volume [L]
        S = 4 * np.pi * r**2 * 1e4  # SV surface area [cm^2]
    else:
        V = p["V"] * 1e15  # Lysosome volume [um^3]
        # V = p["V"]  # Lysosome volume [L]
        S = p["S"]  # Lysosome surface area [cm^2]

    B = (
        p["K_L"]
        + p["Na_L"]
        - init["Cl_L"]
        - init["GLUT_L"]
        + init["H_L"]
        + (p["C_0"] * S) / (V * 1e-15 * p["F"]) * (p["psi_o"] - p["psi_i"])
    )

    Q = (
        V
        * 1e-15
        * (
            p["theta_C"]
            - p["theta"] * (10 ** (-init["pH_L"]) + p["K_L"] + p["Na_L"] + init["Cl_L"])
        )
    )

    p["S"] = S
    p["B"] = B
    p["Q"] = Q
    init["V"] = V
    p["V_0"] = V

    PP = pd.read_excel("datasetProtonPump.xlsx", header=None)
    PP = np.array(PP)
    psi = PP[1:, 0]
    pH = PP[0, 1:]
    values = PP[1:, 1:]
    PP_interp = RegularGridInterpolator(
        points=(psi, pH), values=values, method="linear"
    )
    p["J_VATP"] = PP_interp

    # Ionic species are expressed in concentrations [M]
    # they are converted in number of molecules
    N_A = p["N_A"]
    y0 = (
        init["V"],
        init["pH_L"],
        init["H_L"] * V * 1e-15 * N_A,
        init["Cl_L"] * V * 1e-15 * N_A,
        init["GABA_L"] * V * 1e-15 * N_A,
        init["GLUT_L"] * V * 1e-15 * N_A,
    )
    y0 = np.array(y0)

    return (p, y0)



def extract_solution_SV(y, p):
    N_A = p["N_A"]
    V = y[:, 0]
    V_liters = V * 1e-15
    pH = y[:, 1]
    H = y[:, 2] / V_liters / N_A
    Cl = y[:, 3] / V_liters / N_A
    GABA = y[:, 4]
    GLUT = y[:, 5]

    sol = dict()
    sol["V"] = V
    sol["pH"] = pH
    sol["H"] = H
    sol["Cl"] = Cl
    sol["GABA"] = GABA
    sol["GLUT"] = GLUT

    return sol



def calculate_psi_SV(sol, p):
	## parameters extraction
    F = p["F"]  # Faraday's constant [C / mol]
    N_A = p["N_A"]  # Avogadro constant [mol^-1]

    K = p["K_L"]  # Luminal K+ concentration [M]
    Na = p["Na_L"]  # Luminal Na+ concentration [M]

    C_0 = p["C_0"]  # Lipid bilayer capacitance [F/cm^2]

    psi_o = p["psi_o"]  # Outside leaflet potential [V]
    psi_i = p["psi_i"]  # Inside leaflet potential [V]

    S = p["S"]  # Lysosome surface area [cm^2]
    V_0 = p["V_0"] * 1e-15  # Lysosome initial volume [um^3 to L]

    B = p["B"]  # Concentration of impermeant charges [M]

    V = sol["V"] * 1e-15 # [um^3 to L]
    H = sol["H"]
    Cl = sol["Cl"]
    GABA = sol["GABA"] / V / N_A
    GLUT = sol["GLUT"] / V / N_A
    
    psi = F / (C_0 * S) * (V * (H + K + Na - Cl - GLUT) - B * V_0)
    psi_tot = psi + psi_o - psi_i

    return (psi, psi_tot)










