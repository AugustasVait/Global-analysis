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
        
"""

#%% import libraries and data

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import lmfit



test_data_file = np.loadtxt("D:\\Python Playground\\global analysis test\\test data\\"+
                          "pp_g68_340_1,5uJ_30kHz_eV.dat_pptraces_nochirp.dat")

#%% funkcijos


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
    N_ext, t_00, t_Laser = 2, 0, 0.35
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

    fig_c, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
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

    plt.xlim(1.3, 2.6)

    cbar_ax = fig_c.add_axes([0.85, 0.15, 0.05, 0.6])
    fig_c.colorbar(im2, ax=ax1, cax=cbar_ax)
    fig_c.subplots_adjust(hspace=0)
    fig_c.subplots_adjust(right=0.825)

    fig_c.text(0.04, 0.5, 'Delay (ps)', va='center', ha='center',
             rotation='vertical', fontsize=12)
    fig_c.text(0.5, 0.03, 'Probe energy (eV)', va='center', ha='center',
             fontsize=12)
    fig_c.text(0.85, 0.78, r'   $\Delta$T'+' \n(mOD)', fontsize=12)

CV_color_dict = {0: 'black', 1: 'red', 2: 'blue', 3: 'green', 4: 'purple',
                 5: 'orange'}

def kinetikos_2scale(ps_scale, box_kin, title):
    "docstring"
    fig_k, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                                   gridspec_kw={'width_ratios': [1, 3]})

    title = "Box concentration over time " + title

    for idx in range(len(box_kin[0, :])):
        # print(len(box_kin[0, :]), idx)
        ax1.plot(ps_scale, box_kin[:, idx],
                 color=CV_color_dict[idx])
        ax2.plot(ps_scale, box_kin[:, idx],
                 label='box '+str(idx), color=CV_color_dict[idx])

    ax1.plot(ps_scale, laser_gauss(ps_scale),
                 color='purple')
    ax2.plot(ps_scale, laser_gauss(ps_scale),
                 label='laser ', color='purple')

#    ax1.plot(real_data[0], real_data[1][:, ev_idx], label='sum')
#    ax2.plot(real_data[0], real_data[1][:, ev_idx], label='experimental')

    ax1.set_xlim(-2, 2)

    ax2.set_xlim(2, 7000)
    ax2.set_xscale("log")

    fig_k.text(0.5, 0.85, title, va='center', ha='center', fontsize=12)

    ax1.spines['right'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    # ax1.tick_params(labeltop=False)
    fig_k.subplots_adjust(wspace=0)
    ax2.tick_params(labelleft=False, length=0)
    fig_k.subplots_adjust(top=0.80)
    fig_k.text(0.04, 0.5, 'Concentration (a.u.)', va='center', ha='center',
             rotation='vertical', fontsize=12)
    fig_k.text(0.5, 0.03, 'Delay (ps)', va='center', ha='center', fontsize=12)
    fig_k.legend()



# %%  raw data processing

raw_ev_axis = test_data_file[0,1:]
positive_cutoff = np.searchsorted(test_data_file[1:,0]+0.5,0, side='left')+1
raw_ps_axis = test_data_file[positive_cutoff:,0]+0.5

X = raw_ev_axis
Y = raw_ps_axis

raw_carpet =  test_data_file[positive_cutoff:,1:]

# %% Modeliavimas

modelis = lmfit.Model(box_scheme, independent_vars=['X', 'Y'])



#print('parameter names: {}'.format(modelis.param_names))

modelis.set_param_hint('t_21', value=0.550, min=0, max=1e5, vary=True)
modelis.set_param_hint('t_11', value=50000.0, min=0, max=1e5, vary=True)
modelis.set_param_hint('t_20', value=12.0, min=0, max=1e5, vary=True)
modelis.set_param_hint('t_01', value=5.0, min=0, max=1e5, vary=True)
modelis.set_param_hint('t_03', value=1.0, min=0, max=1e5, vary=True)
modelis.set_param_hint('t_33', value=150.0, min=0, max=1e5, vary=True)

modelis.set_param_hint('box0_a', value=1.0, min=0, max=10000.0, vary=True)
modelis.set_param_hint('box0_xc', value=1.5, min=1.4, max=2.4, vary=True)
modelis.set_param_hint('box0_w', value=1.0, min=0, max=2.0, vary=True)
modelis.set_param_hint('box0_b', value=1.0, min=0, max=1.0, vary=True)
modelis.set_param_hint('box0_y0', value=0, min=-1, max=1.0, vary=False)

modelis.set_param_hint('box1_a', value=1.0, min=0, max=10000.0, vary=True)
modelis.set_param_hint('box1_xc', value=2, min=1.4, max=2.4, vary=True)
modelis.set_param_hint('box1_w', value=1.0, min=0, max=2.00, vary=True)
modelis.set_param_hint('box1_b', value=1.0, min=0, max=1.0, vary=True)
modelis.set_param_hint('box1_y0', value=0, min=-1, max=1.0, vary=False)

modelis.set_param_hint('box2_a', value=0.0, min=0, max=10000.0, vary=False)
modelis.set_param_hint('box2_xc', value=1.5, min=1.4, max=2.4, vary=False)
modelis.set_param_hint('box2_w', value=1.0, min=0, max=2.0, vary=False)
modelis.set_param_hint('box2_b', value=1.0, min=0, max=1.0, vary=False)
modelis.set_param_hint('box2_y0', value=0, min=-1, max=1.0, vary=False)

modelis.set_param_hint('box3_a', value=1.0, min=0, max=10000.0, vary=True)
modelis.set_param_hint('box3_xc', value=2, min=1.4, max=2.4, vary=True)
modelis.set_param_hint('box3_w', value=1.0, min=0, max=2.0, vary=True)
modelis.set_param_hint('box3_b', value=1.0, min=0, max=1.0, vary=True)
modelis.set_param_hint('box3_y0', value=0, min=-1, max=1.0, vary=False)

#modelis.print_param_hints(8)

parametrai = modelis.make_params()

rezultatas = modelis.fit(raw_carpet, X=X, Y=Y, params=parametrai, method='differential_evolution')

print(lmfit.fit_report(rezultatas))

#%% plotavimas

# fig = plt.figure()
# ax0 = fig.add_subplot(1, 1, 1)
# ax0.plot(test_ev_axis, test_gauss0, label='test0')
# ax0.plot(test_ev_axis, test_gauss1, label='test1')
# ax0.plot(test_ev_axis, test_gauss2, label='test2')
# ax0.plot(test_ev_axis, test_gauss3, label='test3')
# fig.legend()
# plt.xlim(1.3, 2.6)

t_21 = rezultatas.best_values['t_21']
t_11 = rezultatas.best_values['t_11']
t_20 = rezultatas.best_values['t_20']
t_01 = rezultatas.best_values['t_01']
t_03 = rezultatas.best_values['t_03']
t_33 = rezultatas.best_values['t_33']

*model_t_const, = t_21, t_11, t_20, t_01, t_03, t_33

model_kinetic = kinetic_solve(Y, model_t_const)

kinetikos_2scale(Y, model_kinetic, 'test')

box0_a = rezultatas.best_values['box0_a']
box0_xc = rezultatas.best_values['box0_xc']
box0_w = rezultatas.best_values['box0_w']
box0_b = rezultatas.best_values['box0_b']
box0_y0 = rezultatas.best_values['box0_y0']

box1_a = rezultatas.best_values['box1_a']
box1_xc = rezultatas.best_values['box1_xc']
box1_w = rezultatas.best_values['box1_w']
box1_b = rezultatas.best_values['box1_b']
box1_y0 = rezultatas.best_values['box1_y0']

box2_a = rezultatas.best_values['box2_a']
box2_xc = rezultatas.best_values['box2_xc']
box2_w = rezultatas.best_values['box2_w']
box2_b = rezultatas.best_values['box2_b']
box2_y0 = rezultatas.best_values['box2_y0']

box3_a = rezultatas.best_values['box3_a']
box3_xc = rezultatas.best_values['box3_xc']
box3_w = rezultatas.best_values['box3_w']
box3_b = rezultatas.best_values['box3_b']
box3_y0 = rezultatas.best_values['box3_y0']

box0_spectra = skewed_g(X, box0_a, box0_xc, box0_w, box0_b, box0_y0)

box1_spectra = skewed_g(X, box1_a, box1_xc, box1_w, box1_b, box1_y0)

box2_spectra = skewed_g(X, box2_a, box2_xc, box2_w, box2_b, box2_y0)

box3_spectra = skewed_g(X, box3_a, box3_xc, box3_w, box3_b, box3_y0)

fig = plt.figure()
ax0 = fig.add_subplot(1, 1, 1)
ax0.plot(X, box0_spectra, label='box 0', color=CV_color_dict[0])
ax0.plot(X, box1_spectra, label='box 1', color=CV_color_dict[1])
ax0.plot(X, box2_spectra, label='box 2', color=CV_color_dict[2])
ax0.plot(X, box3_spectra, label='box 3', color=CV_color_dict[3])
fig.legend()
plt.xlim(1.3, 2.6)


col_low_high_mp = [-1, 20, 'viridis']

carpet_plot(X, Y, rezultatas.best_fit, 'model', col_low_high_mp)

col_low_high_mp = [-1, 20, 'viridis']

carpet_plot(X, Y, raw_carpet, 'raw', col_low_high_mp)

col_low_high_mp = [-3, 3, 'terrain']

carpet_plot(X, Y, raw_carpet-rezultatas.best_fit, 'rezidual', col_low_high_mp)

