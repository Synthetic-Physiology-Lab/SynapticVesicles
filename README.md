# SynapticVesicles
___

A computational framework for modeling presynaptic terminals synaptic vesicle dynamics, pH regulation, ion fluxes and neurotransmitters accumulation with or without GABA (VGAT) or glutamate (VGLUT-1) transporters.

Simulation of synaptic vesicles acidification dynamics with or without GABA (VGAT) or glutamate (VGLUT-1) transporters.

## Overview
___

The SynapticVesicles repository provides mathematical models and simulation tools for investigating the biophysical properties of synaptic vesicles (and lysosomes). The framework focuses on modeling proton pumping, ion transport, excitatory (glutamate) and inhibitory (GABA) neurotransmitters accumulation, membrane potential, and pH dynamics in these organelles.

This repository contains all the scripts that have been used in the paper "**An open-science computational model of organelle acidification to integrate putative mechanisms of synaptic vesicle acidification and filling**", DOI: TO ADD.

## Repository content
___

### Core
The core of this repository is the `models.py` file. It contains all the ODE models implementations that are called in the other scripts to run the simulations and other convenient functions.

### Input configuration files
The `input_**model**.yml` files contain the **model** parameter values and initial conditions.

### Data files
The `.csv` files either contain experimental curves that were used to fit some model parameters or the V-ATPase proton pump acitivity as a function of pH and membrane potential psi.

### Simulation running files
All other files are scripts that run the different models simulations, fit parameters and plot the simulation and fitting results.

## Installation
___

Create a new python virtual environment (recommended python 3.12).

In it, install required dependencies:
```bash
   pip install numpy scipy matplotlib pyyaml pandas lmfit
```

Clone the repository:
```bash
git clone https://github.com/Synthetic-Physiology-Lab/SynapticVesicles.git
cd SynapticVesicles
```

## Google Colab interactive Jupyter notebooks
___

If you prefer, all the different models are also easily accessible in interactive Jupyter notebooks hosted on Google Colab at the following links:
1. [Lysosome](https://colab.research.google.com/drive/1fMFORAUI4OyFXsIHx-d_H8YN3kjAsPT-?usp=drive_link)
2. [Bare SV](https://colab.research.google.com/drive/1EtV4i53Z5IgD20yOeSYRfgFmIQYmw502?usp=drive_link)
3. [GABA SV](https://colab.research.google.com/drive/12dsuN_Rb9_ChnkmmaeYe6zLxL27rClAR?usp=drive_link)
4. [GLUT SV (maycox)](https://colab.research.google.com/drive/1IbTQuPaBykC0HO2LEl6o_AIuEbu8cd93?usp=drive_link)
5. [GLUT SV (kolen)](https://colab.research.google.com/drive/1XpFQSpKi8KrY1Wut5jfovRHGaXPjgVp0?usp=drive_link)

