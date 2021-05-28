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
        ----------+-------------------------------+--------------
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
    Spectra =  A*np.exp(-np.log(2)*((np.log(1+2*b*(X-x0)/W)/b)**2))+Y0
    
    return np.nan_to_num(Spectra)

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
    
    return odeint(box_kinetic_model, N_0, x_data, args=(t_constants,))
    
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

def box_kinetic_model(N, t, t_constants):
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
    
    N_0, N_1, N_2, N_3= N[0], N[1], N[2], N[3]
    
    t_21, t_11, t_20, t_01, t_03, t_33 = t_constants
    
    generation = laser_gauss(t)
    
    dN_2 = generation - N_2/t_21 - N_2/t_20
    dN_1 = N_2/t_21 + N_0/t_01 - N_1/t_11
    dN_0 = N_2/t_20 - N_0/t_01 - N_0/t_03
    dN_3 = N_0/t_03 - N_3/t_33
    
    dNdt = [dN_0, dN_1, dN_2, dN_3] 
    
    return dNdt


def box_scheme(X, Y,
               t_21, t_11, t_20, t_01, t_03, t_33,
               box0_a, box0_xc, box0_w, box0_b, box0_y0,
               box1_a, box1_xc, box1_w, box1_b, box1_y0,
               box2_a, box2_xc, box2_w, box2_b, box2_y0,
               box3_a, box3_xc, box3_w, box3_b, box3_y0):
    
    *t_const, = t_21, t_11, t_20, t_01, t_03, t_33
    
    box_fill = kinetic_solve(Y, t_const)
    
    box0_spectra = skewed_g(X, box0_a, box0_xc, box0_w, box0_b, box0_y0)
    box1_spectra = skewed_g(X, box1_a, box1_xc, box1_w, box1_b, box1_y0)
    box2_spectra = skewed_g(X, box2_a, box2_xc, box2_w, box2_b, box2_y0)
    box3_spectra = skewed_g(X, box3_a, box3_xc, box3_w, box3_b, box3_y0)

    box0_component = np.outer(box0_spectra, box_fill[:,0]).T
    box1_component = np.outer(box1_spectra, box_fill[:,1]).T
    box2_component = np.outer(box2_spectra, box_fill[:,2]).T
    box3_component = np.outer(box3_spectra, box_fill[:,3]).T

    return box0_component + box1_component + box2_component + box3_component



def carpet_plot(x_axis, y_axis, z_data, title, clr_min_max_mp):
    """docstings"""

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                   gridspec_kw={'height_ratios': [3, 1]})

    im1 = ax1.pcolormesh(x_axis, y_axis, z_data, shading='auto',
                         cmap=clr_min_max_mp[2])
    im1.set_clim(vmin=clr_min_max_mp[0], vmax=clr_min_max_mp[1])
    im2 = ax2.pcolormesh(x_axis, y_axis, z_data, shading='auto',
                         cmap=clr_min_max_mp[2])
    im2.set_clim(vmin=clr_min_max_mp[0], vmax=clr_min_max_mp[1])

    ax1.set_ylim(2, 7000)
    ax1.set_yscale("log")
    ax2.set_ylim(-2, 2)

    ax1.set_title(title)

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)

    plt.xlim(1, 3)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.6])
    fig.colorbar(im2, ax=ax1, cax=cbar_ax)
    fig.subplots_adjust(hspace=0)
    fig.subplots_adjust(right=0.825)

    fig.text(0.04, 0.5, 'Delay (ps)', va='center', ha='center',
             rotation='vertical', fontsize=12)
    fig.text(0.5, 0.03, 'Probe energy (eV)', va='center', ha='center',
             fontsize=12)
    fig.text(0.85, 0.78, r'   $\Delta$T'+' \n(mOD)', fontsize=12)

CV_color_dict = {0: 'black', 1: 'red', 2: 'blue', 3: 'green', 4: 'purple',
                 5: 'orange'}

def kinetikos_2scale(ps_scale, box_kin, labels, title):
    "docstring"
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                                   gridspec_kw={'width_ratios': [1, 3]})

    title = "Box concentration over time " + title

    for idx in range(len(box_kin[0, :])):
        # print(len(box_kin[0, :]), idx)
        ax1.plot(ps_scale, box_kin[:, idx],
                 color=CV_color_dict[idx])
        ax2.plot(ps_scale, box_kin[:, idx],
                 label='box '+str(idx), color=CV_color_dict[idx])


#    ax1.plot(real_data[0], real_data[1][:, ev_idx], label='sum')
#    ax2.plot(real_data[0], real_data[1][:, ev_idx], label='experimental')

    ax1.set_xlim(-2, 2)

    ax2.set_xlim(2, 7000)
    ax2.set_xscale("log")

    fig.text(0.5, 0.85, title, va='center', ha='center', fontsize=12)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    # ax1.tick_params(labeltop=False)
    fig.subplots_adjust(wspace=0)
    ax2.tick_params(labelleft=False, length=0)
    fig.subplots_adjust(top=0.80)
    fig.text(0.04, 0.5, 'Concentration (a.u.)', va='center', ha='center',
             rotation='vertical', fontsize=12)
    fig.text(0.5, 0.03, 'Delay (ps)', va='center', ha='center', fontsize=12)
    fig.legend()

#%% test calc
test_ev_axis = np.linspace(1,3,1024)
test_ps_axis = np.append(np.linspace(0,1,10), np.logspace(0,4,1014, base=10))

# t_21, t_11, t_20, t_01, t_03, t_33

*test_t_const, = 10, 5000, 50, 400, 50, 6000

test_kinetic = kinetic_solve(test_ps_axis, test_t_const)

#A, x0, W, b, Y0

tbox0_a, tbox0_xc, tbox0_w, tbox0_b, tbox0_y0 = 10, 2, 0.33, 0, 0
tbox1_a, tbox1_xc, tbox1_w, tbox1_b, tbox1_y0 = 9, 2.3, 0.44, 0, 0
tbox2_a, tbox2_xc, tbox2_w, tbox2_b, tbox2_y0 = 60, 1.8, 0.44, 0.5, 0
tbox3_a, tbox3_xc, tbox3_w, tbox3_b, tbox3_y0 = 5, 2, 0.5, 0, 0

test_gauss0 = skewed_g(test_ev_axis, 
                       tbox0_a, tbox0_xc, tbox0_w, tbox0_b, tbox0_y0)

test_gauss1 = skewed_g(test_ev_axis, 
                       tbox1_a, tbox1_xc, tbox1_w, tbox1_b, tbox1_y0)

test_gauss2 = skewed_g(test_ev_axis, 
                       tbox2_a, tbox2_xc, tbox2_w, tbox2_b, tbox2_y0)

test_gauss3 = skewed_g(test_ev_axis, 
                       tbox3_a, tbox3_xc, tbox3_w, tbox3_b, tbox3_y0)

test_carpet = box_scheme(test_ev_axis, test_ps_axis,
           10, 5000, 50, 400, 50, 6000,
           tbox0_a, tbox0_xc, tbox0_w, tbox0_b, tbox0_y0,
           tbox1_a, tbox1_xc, tbox1_w, tbox1_b, tbox1_y0,
           tbox2_a, tbox2_xc, tbox2_w, tbox2_b, tbox2_y0,
           tbox3_a, tbox3_xc, tbox3_w, tbox3_b, tbox3_y0)

#%%plotavimas

fig = plt.figure()
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(test_ev_axis, test_gauss0, label='test0')
ax0.plot(test_ev_axis, test_gauss1, label='test1')
ax0.plot(test_ev_axis, test_gauss2, label='test2')
ax0.plot(test_ev_axis, test_gauss3, label='test3')
fig.legend()

kinetikos_2scale(test_ps_axis, test_kinetic, '0123', 'test')

col_low_high_mp = [-1, 15, 'viridis']

carpet_plot(test_ev_axis, test_ps_axis, test_carpet, 'test', col_low_high_mp)