# -*- coding: utf-8 -*-
"""
Created on Thu May 27 22:21:11 2021

@author: augus

Global analysis w/ LMfit

Objectives:
    
    uzrasyti modeline funkcija, Modelinė f-ja susideda iš:     
        +kreivu gauso spektru (energijos dimensija)
        +kinetiku (dif lygčiu sprendiniu) (laiko dimensija)
        jas sudauginame ir susumuojame
        
    vizualizacijos sketch:
        spektrai  | spektrai * kinetikos karpetas | eksperimentas
        ---------------------------------------------------------
        kinetikos | spektrai * kinetikos pjūvis   | residual
"""

#%% imports and definitions

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def skewed_g(X, A, x0, W, b, Y0):
    """
    pagal frasier1969

    Parameters
    ----------
    X : array
        Energy scale.
    A : float
        Amplitude.
    x0 : float
        Spectral center.
    W : float
        Spectra width.
    b : float
        skewednes.
    Y0 : float
        baseline.

    Returns
    -------
    array
        same dimensions as X.

    """
    if b == 0:
        b = 1e-10
    return A*np.exp(-np.log(2)*((np.log(1+2*b*(X-x0)/W)/b)**2))+Y0

def kinetic_solve(x_data, t_constants):
    """
    funtion to solve ODE system according to global analysis box scheme
    

    Parameters
    ----------
    X : aray
        time scale.
    t_constants : list
        list of t constants.
        
    Returns
    -------
    list of arrays same shape as X.

    """
    
    N_0 = [0, 0, 0, 0] #initial values
    
    return odeint(box_scheme_model, N_0, x_data, args=(t_constants,))
    
def laser_gauss (t):
    """
    parametrai nurodomi funkcijoje.

    Parameters
    ----------
    t : array
        time axis.

    Returns
    -------
    float
        lazerio intensyvumas duotu laiku.

    """
    N_ext, t_00, t_Laser = 1, 1, 0.35
    return (N_ext)*np.exp(-(t-t_00)**2/(t_Laser**2))

def box_scheme_model(N, t, t_constants):
    """
    pagal scipy.odeint pvz
    
    ODE aprašančios dėžučių schemą

    Parameters
    ----------
    N : list
        dif lygties kintamieji.
    t : array
        laiko ašis.
    t_constants : list
        laikinės konstantos.

    Returns
    -------
    dNdt : list
        kintamųjų pokytis.

    """
    
    N_5d2, N_5d1, N_ST, N_NR= N[0], N[1], N[2], N[3]
    
    t_21, t_1G, t_2st, t_st1, t_stNR, t_NRG = t_constants
    
    generation = laser_gauss(t)
    
    dN_5d2 = generation -N_5d2/t_21 - N_5d2/t_2st
    dN_5d1 = N_5d2/t_21 + N_ST/t_st1 - N_5d1/t_1G
    dN_ST = N_5d2/t_2st - N_ST/t_st1 - N_ST/t_stNR
    dN_NR = N_ST/t_stNR - N_NR/t_NRG
    
    dNdt = [dN_5d2, dN_5d1, dN_ST, dN_NR] 
    
    return dNdt
    
#%% test calc
test_ev_axis = np.linspace(1,3,1024)
test_ps_axis = np.logspace(-3,4,1024, base=10)

*test_t_const, = 1, 100000, 300, 400, 500, 6000

test_gauss = skewed_g(test_ev_axis, 10, 2, 0.33, 0, 0)
test_kinetic = kinetic_solve(test_ps_axis, test_t_const)

#%%plotavimas

fig = plt.figure()
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(test_ev_axis, test_gauss, label='test')
