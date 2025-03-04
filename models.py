import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator


def create_VATP_interp_obj(VATP_grid_file):
    VATP = pd.read_csv(VATP_grid_file, header=None)
    VATP = np.array(VATP)
    psi = VATP[1:, 0]
    pH = VATP[0, 1:]
    values = VATP[1:, 1:]
    VATP_interp = RegularGridInterpolator(
        points=(psi, pH), values=values, method="linear"
    )
    return VATP_interp


def lysosome_model(y: np.ndarray, t: np.ndarray, p: dict):
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
    #print(V, psi)
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
    dV = J_W * v_W / 1e6 * 1e15
    dpH = -(N_V * J_V + N_ClC * J_ClC + J_H) / beta / V / N_A
    dH = N_V * J_V + ClC_H * N_ClC * J_ClC + J_H
    dK = J_K
    dNa = J_Na
    dCl = J_Cl - ClC_Cl * N_ClC * J_ClC

    dy = (dV, dpH, dH, dK, dNa, dCl)
    return np.array(dy)


def set_lysosome_model(p, init, VATP_grid_file="datasetProtonPump.csv"):
    if p["d"] > 0:
        r = p["d"] * 1e-6 / 2  # Lysosome radius [m]
        V = 4 / 3 * np.pi * r**3 * 1e18  # SV volume [um^3]
        # V = 4 / 3 * np.pi * r**3 * 1e3  # Lysosome volume [L]
        S = 4 * np.pi * r**2 * 1e4  # Lysosome surface area [cm^2]
    else:
        V = init["V"] * 1e15  # Lysosome volume [um^3]
        S = p["S"]  # Lysosome surface area [cm^2]

    B = (
        init["K_L"]
        + init["Na_L"]
        - init["Cl_L"]
        + init["H_L"]
        + (p["C_0"] * S) / (V * 1e-15 * p["F"]) * (p["psi_o"] - p["psi_i"] - p["psi_tot"])
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
    p["J_VATP"] = create_VATP_interp_obj(VATP_grid_file)

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


def extract_solution_lysosome(y, p):
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


def calculate_psi_lysosome(sol, p):
    ## parameters extraction
    F = p["F"]  # Faraday's constant [C / mol]

    C_0 = p["C_0"]  # Lipid bilayer capacitance [F/cm^2]

    psi_o = p["psi_o"]  # Outside leaflet potential [V]
    psi_i = p["psi_i"]  # Inside leaflet potential [V]

    S = p["S"]  # Lysosome surface area [cm^2]
    V_0 = p["V_0"] * 1e-15  # Lysosome initial volume [um^3 to L]

    B = p["B"]  # Concentration of impermeant charges [M]

    V = sol["V"] * 1e-15  # [um^3 to L]
    H = sol["H"]
    K = sol["K"]
    Na = sol["Na"]
    Cl = sol["Cl"]

    psi = F / (C_0 * S) * (V * (H + K + Na - Cl) - B * V_0)
    psi_tot = psi + psi_o - psi_i

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

    # ClC-7 pump stoichiometry
    ClC_Cl = p["ClC_Cl"]  # ClC-7 Cl- Stoichiometry
    ClC_H = p["ClC_H"]  # ClC-7 H+ Stoichiometry

    ## state variables extraction
    V, pH, H, Cl = y  # Ionic species are expressed in number of molecules
    # they are converted in concentrations [M]
    V = V * 1e-15  # [um^3 to L]
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
    dV = J_W * v_W / 1e6 * 1e15  # [L/s to um^3/s]
    dpH = -(N_V * J_V + ClC_H * N_ClC * J_ClC + J_H) / beta / V / N_A
    dH = N_V * J_V + ClC_H * N_ClC * J_ClC + J_H
    dCl = -ClC_Cl * N_ClC * J_ClC  # + J_Cl

    dy = (dV, dpH, dH, dCl)
    return np.array(dy)


def set_bareSV_model(p, init, VATP_grid_file="datasetProtonPump.csv"):
    if p["d"] > 0:
        r = p["d"] * 1e-6 / 2  # SV radius [m]
        V = 4 / 3 * np.pi * r**3 * 1e18  # SV volume [um^3]
        # V = 4 / 3 * np.pi * r**3 * 1e3  # SV volume [L]
        S = 4 * np.pi * r**2 * 1e4  # SV surface area [cm^2]
    else:
        V = init["V"] * 1e15  # Lysosome volume [um^3]
        # V = p["V"]  # Lysosome volume [L]
        S = p["S"]  # Lysosome surface area [cm^2]

    B = (
        p["K_L"]
        + p["Na_L"]
        - init["Cl_L"]
        + init["H_L"]
        + (p["C_0"] * S) / (V * 1e-15 * p["F"]) * (p["psi_o"] - p["psi_i"] - p["psi_tot"])
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
    p["J_VATP"] = create_VATP_interp_obj(VATP_grid_file)

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

    V = sol["V"] * 1e-15  # [um^3 to L]
    H = sol["H"]
    Cl = sol["Cl"]

    psi = F / (C_0 * S) * (V * (H + K + Na - Cl) - B * V_0)
    psi_tot = psi + psi_o - psi_i

    return (psi, psi_tot)


def calculate_Hflow_bareSV(sol, p):

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
    V = sol["V"] * 1e-15  # [um^3 to L]
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
            gg.append(U[i] / (1 - np.exp(-U[i])))
        else:
            gg.append(
                1 / (1 - U[i] / 2 + U[i] ** 2 / 6 - U[i] ** 3 / 24 + U[i] ** 4 / 120)
            )
    gg = np.array(gg)

    J_H = P_H * S * gg * (10 ** (-pHe) * np.exp(-U) - 10 ** (-pHi)) * N_A / 1000

    Hflow = dict()
    Hflow["vATPase"] = N_V * J_V
    Hflow["ClC3"] = ClC_H * N_ClC * J_ClC
    Hflow["leak"] = J_H

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
    N_VGAT = p["N_VGAT"]  # Number of GABA transporters VGAT
    N_VGLUT = p["N_VGLUT"]  # Number of glutamate transporters VGLUT-1

    # ClC-7 pump stoichiometry
    ClC_Cl = p["ClC_Cl"]  # ClC-7 Cl- Stoichiometry
    ClC_H = p["ClC_H"]  # ClC-7 H+ Stoichiometry

    # VGAT transporter stoichiometry
    VGAT_GABA = p["VGAT_GABA"]  # VGAT GABA Stoichiometry
    VGAT_H = p["VGAT_H"]  # VGAT H+ Stoichiometry

    # VGLUT-1 transporter stoichiometry
    VGLUT_GLUT = p["VGLUT_GLUT"]  # VGLUT gluatamate Stoichiometry
    VGLUT_H = p["VGLUT_H"]  # VGLUT H+ Stoichiometry

    # Neurotransmitters trasnsport rates
    k_GABA = p["k_GABA"]  # GABA transport rate [s^-1]
    k_GLUT = p["k_GLUT"]  # Glutamate transport rate [s^-1]
    tau_VGAT = p["tau_VGAT"]  # VGAT time constant [s]
    tau_GABA = p["tau_GABA"]  # GABA efflux time constant [s]
    tau_VGLUT = p["tau_VGLUT"]  # VGLUT-1 efflux time constant [s]
    P_Cl_VGLUT = p["P_Cl_VGLUT"]  # VGLUT-1 Cl- permeability [s^-1]

    ## state variables extraction
    V, pH, H, Cl, GABA, GLUT = y  # Ionic species are expressed in number of molecules
    # they are converted in concentrations [M]
    V = V * 1e-15  # [um^3 to L]
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
    J_Cl_VGLUT = N_VGLUT * P_Cl_VGLUT * 1e-10 * gg * (Cle - Cli * np.exp(-U)) * N_A / 1000
    J_W = P_W * S * (theta * (10 ** (-pH) + K + Na + Cl) + Q / V - theta_C)

    if tau_VGAT == 0:
        J_GABAin = k_GABA
    elif tau_VGAT > 0:
        J_GABAin = k_GABA * (1 - np.exp(-t / tau_VGAT))
    
    if tau_GABA == 0:
        J_GABAout = 0
    elif tau_GABA > 0:
        J_GABAout = GABA * V * N_A / tau_GABA

    J_GLUTin = k_GLUT
    
    if tau_VGLUT == 0:
        J_GLUTout = 0
    elif tau_VGLUT > 0:
        J_GLUTout = GLUT * V * N_A / tau_VGLUT

    ## derivatives calculation
    H_tot = (
        N_V * J_V
        + ClC_H * N_ClC * J_ClC
        + J_H
        - VGAT_H * N_VGAT * J_GABAin
        - VGLUT_H * N_VGLUT * (J_GLUTin - J_GLUTout)
    )
    dV = J_W * v_W / 1e6 * 1e15  # [L/s to um^3/s]
    dpH = -H_tot / beta / V / N_A
    dH = H_tot
    dCl = -ClC_Cl * N_ClC * J_ClC + J_Cl_VGLUT
    dGABA = VGAT_GABA * N_VGAT * J_GABAin - J_GABAout
    dGLUT = VGLUT_GLUT * N_VGLUT * (J_GLUTin - J_GLUTout)

    dy = (dV, dpH, dH, dCl, dGABA, dGLUT)
    return np.array(dy)


def SV_model_modified(y: np.ndarray, t: np.ndarray, p: dict):
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
    N_VGAT = p["N_VGAT"]  # Number of GABA transporters VGAT
    N_VGLUT = p["N_VGLUT"]  # Number of glutamate transporters VGLUT-1

    # ClC-7 pump stoichiometry
    ClC_Cl = p["ClC_Cl"]  # ClC-7 Cl- Stoichiometry
    ClC_H = p["ClC_H"]  # ClC-7 H+ Stoichiometry

    # VGAT transporter stoichiometry
    VGAT_GABA = p["VGAT_GABA"]  # VGAT GABA Stoichiometry
    VGAT_H = p["VGAT_H"]  # VGAT H+ Stoichiometry

    # VGLUT-1 transporter stoichiometry
    VGLUT_GLUT = p["VGLUT_GLUT"]  # VGLUT gluatamate Stoichiometry
    VGLUT_H = p["VGLUT_H"]  # VGLUT H+ Stoichiometry

    # Neurotransmitters trasnsport rates
    k_GABA = p["k_GABA"]  # GABA transport rate [s^-1]
    k_GLUT = p["k_GLUT"]  # Glutamate transport rate [s^-1]
    tau_VGAT = p["tau_VGAT"]  # VGAT time constant [s]
    tau_GABA = p["tau_GABA"]  # GABA efflux time constant [s]
    tau_VGLUT = p["tau_VGLUT"]  # VGLUT-1 efflux time constant [s]
    P_Cl_VGLUT = p["P_Cl_VGLUT"]  # VGLUT-1 Cl- permeability [s^-1]

    ## state variables extraction
    V, pH, H, Cl, GABA, GLUT = y  # Ionic species are expressed in number of molecules
    # they are converted in concentrations [M]
    V = V * 1e-15  # [um^3 to L]
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
    J_Cl_VGLUT = N_VGLUT * P_Cl_VGLUT * 1e-10 * gg * (Cle - Cli * np.exp(-U)) * N_A / 1000
    J_W = P_W * S * (theta * (10 ** (-pH) + K + Na + Cl) + Q / V - theta_C)

    if tau_VGAT == 0:
        J_GABAin = k_GABA
    elif tau_VGAT > 0:
        J_GABAin = k_GABA * (1 - np.exp(-t / tau_VGAT)) * (1 / (1 + np.exp(-10 * (10000 - GABA * V * N_A))))
    
    if tau_GABA == 0:
        J_GABAout = 0
    elif tau_GABA > 0:
        J_GABAout = GABA * V * N_A / tau_GABA

    J_GLUTin = k_GLUT
    
    if tau_VGLUT == 0:
        J_GLUTout = 0
    elif tau_VGLUT > 0:
        J_GLUTout = GLUT * V * N_A / tau_VGLUT

    ## derivatives calculation
    H_tot = (
        N_V * J_V
        + ClC_H * N_ClC * J_ClC
        + J_H
        - VGAT_H * N_VGAT * J_GABAin
        - VGLUT_H * N_VGLUT * (J_GLUTin - J_GLUTout)
    )
    dV = J_W * v_W / 1e6 * 1e15  # [L/s to um^3/s]
    dpH = -H_tot / beta / V / N_A
    dH = H_tot
    dCl = -ClC_Cl * N_ClC * J_ClC + J_Cl_VGLUT
    dGABA = VGAT_GABA * N_VGAT * J_GABAin - J_GABAout
    dGLUT = VGLUT_GLUT * N_VGLUT * (J_GLUTin - J_GLUTout)

    dy = (dV, dpH, dH, dCl, dGABA, dGLUT)
    return np.array(dy)


def set_SV_model(p, init, VATP_grid_file="datasetProtonPump.csv"):
    if p["d"] > 0:
        r = p["d"] * 1e-6 / 2  # SV radius [m]
        V = 4 / 3 * np.pi * r**3 * 1e18  # SV volume [um^3]
        # V = 4 / 3 * np.pi * r**3 * 1e3  # SV volume [L]
        S = 4 * np.pi * r**2 * 1e4  # SV surface area [cm^2]
    else:
        V = init["V"] * 1e15  # Lysosome volume [um^3]
        # V = p["V"]  # Lysosome volume [L]
        S = p["S"]  # Lysosome surface area [cm^2]

    B = (
        p["K_L"]
        + p["Na_L"]
        - init["Cl_L"]
        - init["GLUT_L"]
        + init["H_L"]
        + (p["C_0"] * S) / (V * 1e-15 * p["F"]) * (p["psi_o"] - p["psi_i"] - p["psi_tot"])
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
    p["J_VATP"] = create_VATP_interp_obj(VATP_grid_file)

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

    V = sol["V"] * 1e-15  # [um^3 to L]
    H = sol["H"]
    Cl = sol["Cl"]
    GABA = sol["GABA"] / V / N_A
    GLUT = sol["GLUT"] / V / N_A

    psi = F / (C_0 * S) * (V * (H + K + Na - Cl - GLUT) - B * V_0)
    psi_tot = psi + psi_o - psi_i

    return (psi, psi_tot)
