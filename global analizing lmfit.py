"""
Created on Thu May 27 22:21:11 2021.

@author: augus

Global analysis w/ LMfit

ToDo:
    Outline view spyderyje.
    Praturtinti komentarais
    Pridėti vaizdinimą kur kitenikos sudaugintos su vidutine spektro verte 
    ir suma palyginta su eksperimentu

"""

# %% import libraries and data

import numpy as np
from scipy.integrate import odeint, quad
import lmfit

import glob_ana_plot
# mano biblioteka su plotinimo funkcijomis, laikoma tame pačiame folderyje


test_data_file = np.loadtxt("C:\\VU FNI Cloud\\CarpetView Modeling\\GYAGG CV\\" +
                            "4box models\\All F\\Final for paper\\" +
                            "pp_g68_340_1,5uJ_30kHz_eV." +
                            "dat_pptraces_nochirp.dat")

# %% funkcijos


def skewed_g(X, A, x0, W, b, Y0):
    """
    Pagal frasier1969.

    Parameters
    ----------
    X : array, Energy scale.
    A : float, Amplitude.
    x0 : float, Spectral center.
    W : float, Spectra width.
    b : float, skewednes.
    Y0 : float, baseline.

    Returns
    -------
    array, same dimensions as X.
    """
    if b == 0:
        b = 1e-10
    Spectra = A*np.exp(-np.log(2)*((np.log(1+2*b*(X-x0)/W)/b)**2))+Y0

    return np.nan_to_num(Spectra)


def kinetic_solve(x_data, t_constants):
    """
    Funtion to solve ODE system according to global analysis box scheme.

    Parameters
    ----------
    X : aray, time scale.
    t_constants : list, list of t constants.

    Returns
    -------
    list of arrays same shape as X.

    """
    N_0 = [0, 0, 0, 0]  # initial values

    return odeint(box_kinetic_model, N_0, x_data, args=(t_constants,))


def laser_normalization(x, w):
    """Nesugalvoju kaip gudriau padaryti.

    Funkcija lazerio integravimui, kad būtų galima normuoti.
    """
    return np.exp(-(x)**2/(w**2))


def laser_gauss(t):
    """
    Parametrai nurodomi funkcijoje.

    Parameters
    ----------
    t : array, time axis.

    Returns
    -------
    float, lazerio intensyvumas duotu laiku.

    """
    # N_ext, t_00, t_Laser = 2, 0.1, 0.35

    laser_w = 0.35
    t_00 = .5

    normalization = quad(laser_normalization, -10, 10, args=laser_w)[0]

    return np.exp(-(t-t_00)**2/(laser_w**2)) / normalization


def box_kinetic_model(N, t, t_constants):
    """
    Pagal scipy.odeint pvz.

    ODE aprašančios dėžučių schemą

    Parameters
    ----------
    N : list, dif lygties kintamieji.
    t : array, laiko ašis.
    t_constants : list, laikinės konstantos.

    Returns
    -------
    dNdt : list,  kintamųjų pokytis.

    """
    N_0, N_1, N_2, N_3 = N[0], N[1], N[2], N[3]

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
    """Modelis."""
    *t_const, = t_21, t_11, t_20, t_01, t_03, t_33

    box_fill = kinetic_solve(Y, t_const)

    box0_spectra = skewed_g(X, box0_a, box0_xc, box0_w, box0_b, box0_y0)
    box1_spectra = skewed_g(X, box1_a, box1_xc, box1_w, box1_b, box1_y0)
    box2_spectra = skewed_g(X, box2_a, box2_xc, box2_w, box2_b, box2_y0)
    box3_spectra = skewed_g(X, box3_a, box3_xc, box3_w, box3_b, box3_y0)

    box0_component = np.nan_to_num(np.outer(box0_spectra, box_fill[:, 0]).T)
    box1_component = np.nan_to_num(np.outer(box1_spectra, box_fill[:, 1]).T)
    box2_component = np.nan_to_num(np.outer(box2_spectra, box_fill[:, 2]).T)
    box3_component = np.nan_to_num(np.outer(box3_spectra, box_fill[:, 3]).T)

    return box0_component + box1_component + box2_component + box3_component


# %%  raw data processing

raw_ev_axis = test_data_file[0, 1:]
positive_cutoff = np.searchsorted(test_data_file[1:, 0]+0.5, 0, side='left')+1
raw_ps_axis = test_data_file[positive_cutoff:, 0]+0.5

X = raw_ev_axis
Y = raw_ps_axis

raw_carpet = test_data_file[positive_cutoff:, 1:]

# %% Modeliavimas

modelis = lmfit.Model(box_scheme, independent_vars=['X', 'Y'])


# print('parameter names: {}'.format(modelis.param_names))

modelis.set_param_hint('t_21', value=0.550, min=0, max=1e5, vary=True)
modelis.set_param_hint('t_11', value=50000.0, min=0, max=1e5, vary=True)
modelis.set_param_hint('t_20', value=12.0, min=0, max=1e5, vary=True)
modelis.set_param_hint('t_01', value=5.0, min=0, max=1e5, vary=True)
modelis.set_param_hint('t_03', value=1.0, min=0, max=1e5, vary=True)
modelis.set_param_hint('t_33', value=150.0, min=0, max=1e5, vary=True)

modelis.set_param_hint('box0_a', value=100.0, min=0, max=10000.0, vary=True)
modelis.set_param_hint('box0_xc', value=1.85, min=1.4, max=2.4, vary=True)
modelis.set_param_hint('box0_w', value=0.6, min=0, max=2.0, vary=True)
modelis.set_param_hint('box0_b', value=0.50, min=-1, max=1.0, vary=True)
modelis.set_param_hint('box0_y0', value=5, min=-10, max=10.0, vary=False)

modelis.set_param_hint('box1_a', value=15.0, min=0, max=10000.0, vary=True)
modelis.set_param_hint('box1_xc', value=1.45, min=1.4, max=1.5, vary=True)
modelis.set_param_hint('box1_w', value=0.80, min=0, max=1.00, vary=True)
modelis.set_param_hint('box1_b', value=-0.166, min=-1, max=1.0, vary=True)
modelis.set_param_hint('box1_y0', value=3.5, min=-10, max=1.00, vary=False)

modelis.set_param_hint('box2_a', value=0.0, min=0, max=10000.0, vary=False)
modelis.set_param_hint('box2_xc', value=1.5, min=1.4, max=2.4, vary=False)
modelis.set_param_hint('box2_w', value=1.0, min=0, max=2.0, vary=False)
modelis.set_param_hint('box2_b', value=1.0, min=0, max=1.0, vary=False)
modelis.set_param_hint('box2_y0', value=0, min=-1, max=1.0, vary=False)

modelis.set_param_hint('box3_a', value=80.0, min=0, max=10000.0, vary=True)
modelis.set_param_hint('box3_xc', value=1.7, min=1.4, max=2.4, vary=True)
modelis.set_param_hint('box3_w', value=0.8, min=0, max=2.0, vary=True)
modelis.set_param_hint('box3_b', value=0, min=-1, max=1.0, vary=True)
modelis.set_param_hint('box3_y0', value=11, min=-1, max=15.0, vary=False)

# modelis.print_param_hints(8)

parametrai = modelis.make_params()

rezultatas = modelis.fit(raw_carpet, X=X, Y=Y, params=parametrai,
                         method='leastsq')

print(lmfit.fit_report(rezultatas))

# %% plotavimas

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

laser = laser_gauss(Y)

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

TA_spectra = np.c_[box0_spectra, box1_spectra, box2_spectra, box3_spectra]

glob_ana_plot.TA_spectral('title',
                          X, [1.3, 2.6],
                          TA_spectra, 
                          [0.01,1.1*np.amax(TA_spectra)], 'linear',
                          ['Box 0', 'Box 1', 'Box 2', 'Box 3'])

glob_ana_plot.TA_spectral('title',
                          X, [1.3, 2.6],
                          TA_spectra, 
                          [0.01,2*np.amax(TA_spectra)], 'log',
                          ['Box 0', 'Box 1', 'Box 2', 'Box 3'])


glob_ana_plot.kinetikos_2scale('title',
                               Y, [-1, 7000], 5,
                               np.c_[laser_gauss(Y), model_kinetic],
                               [0, 1.1*np.amax(model_kinetic)],
                               ['Laser',
                                'box 0', 'box 1', 'box 2', 'box 3', 'box 4'])

glob_ana_plot.carpet_plot('Model',
                          X, [1.3, 2.6],
                          Y, [-1, 7000], 5,
                          rezultatas.best_fit, [-1, 20, 'viridis'])

glob_ana_plot.carpet_plot('Model',
                          X, [1.3, 2.6],
                          Y, [-1, 7000], 5,
                          raw_carpet, [-1, 20, 'viridis'])

glob_ana_plot.carpet_plot('Model',
                          X, [1.3, 2.6],
                          Y, [-1, 7000], 5,
                          raw_carpet-rezultatas.best_fit, [-3, 3, 'terrain'])

import winsound
frequency = 500  # Set Frequency To 2500 Hertz
duration = 1000  # Set Duration To 1000 ms == 1 second
winsound.Beep(frequency, duration)
