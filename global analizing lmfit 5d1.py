"""
Created on Thu May 27 22:21:11 2021.

@author: augus

Global analysis w/ LMfit

To Do:
    Sukurti weights matricą (turbūt loadinant duomenis)
    Outline view spyderyje.
    Praturtinti komentarais
    išvedimas
"""

# %% import libraries and data

import numpy as np
from scipy.integrate import odeint, quad
import lmfit

import glob_ana_plot
# mano biblioteka su plotinimo funkcijomis, laikoma tame pačiame folderyje

import time
t_start = time.time()


# test_data_file = np.loadtxt("C:\\Users\\Lenovo\\Documents\\GitHub\\"+
#                             "Global-analysis\\test data\\"+
#                             "pp_g68_340_1,5uJ_30kHz_eV."+
#                             "dat_pptraces_nochirp.dat")

# test_data_file = np.loadtxt("D:\\Python Playground\\git\\" +
#                             "Global-analysis\\test data\\" +
#                             "pp_g68_340_1,5uJ_30kHz_eV." +
#                             "dat_pptraces_nochirp.dat")

data_file = np.loadtxt("D:\\VU cloud\\Grupės rezultatai\\Scintillators\\" +
                       "GAGG\\GAGG vs Mg vs T PPwlc TCSPC\\n2\\" +
                       "pump 443nm\\pp_N2_443nm_0,8uJ_30kHz_T438K_avg")

test_data_file = data_file
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
    N_0 = [0]  # initial values

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
    N_1 = N[0]

    t_11 = t_constants

    generation = laser_gauss(t)

    dN_1 = generation - N_1/t_11

    dNdt = [dN_1]

    return dNdt


def box_scheme(X, Y,
               t_11,
               box1_a, box1_xc, box1_w, box1_b, box1_y0):
    """Modelis."""
    t_const = t_11

    box_fill = kinetic_solve(Y, t_const)

    box1_spectra = skewed_g(X, box1_a, box1_xc, box1_w, box1_b, box1_y0)

    box1_component = np.nan_to_num(np.outer(box1_spectra, box_fill[:, 0]).T)

    return box1_component


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

modelis.set_param_hint('t_11', value=50000.0, min=0, max=1e5, vary=True)

modelis.set_param_hint('box1_a', value=15.0, min=0, max=20.0, vary=True)
modelis.set_param_hint('box1_xc', value=800, min=400, max=1200, vary=True)
modelis.set_param_hint('box1_w', value=200, min=0, max=1000.00, vary=True)
modelis.set_param_hint('box1_b', value=-0.166, min=-1, max=1.0, vary=True)
modelis.set_param_hint('box1_y0', value=3.5, min=-10, max=10.00, vary=True)


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

t_11 = rezultatas.best_values['t_11']

model_t_const = t_11

model_kinetic = kinetic_solve(Y, model_t_const)

laser = laser_gauss(Y)

box1_a = rezultatas.best_values['box1_a']
box1_xc = rezultatas.best_values['box1_xc']
box1_w = rezultatas.best_values['box1_w']
box1_b = rezultatas.best_values['box1_b']
box1_y0 = rezultatas.best_values['box1_y0']

box1_spectra = skewed_g(X, box1_a, box1_xc, box1_w, box1_b, box1_y0)


TA_spectra = np.c_[box1_spectra]
TA_means = np.mean(TA_spectra, axis=0)

placeholder = []

for spectra in TA_spectra.T:
    if sum(spectra) == 0:
        placeholder.append(spectra)
    else:
        spectra = spectra-np.min(spectra)
        spectra = spectra/np.max(spectra)
        placeholder.append(spectra)
    TA_spectra_norm = np.c_[placeholder].T

TA_kinetic = model_kinetic*TA_means
TA_kinetic_sum = np.sum(TA_kinetic, axis=1)

TA_kinetic_experimental = np.mean(raw_carpet, axis=1)
TA_spectra_experimental = np.c_[np.mean(raw_carpet, axis=0)]

graph1 = glob_ana_plot.TA_spectral('TA Spectra experimental',
                                   X, [400, 1200],
                                   TA_spectra_experimental,
                                   [1, 2*np.amax(TA_spectra_experimental)], 'log',
                                   ['experimental'])


graph1 = glob_ana_plot.TA_spectral('TA Spectra',
                                   X, [400, 1200],
                                   TA_spectra,
                                   [1, 2*np.amax(TA_spectra)], 'log',
                                   ['Box 0', 'Box 1', 'Box 2', 'Box 3'])

graph2 = glob_ana_plot.TA_spectral('Normalized TA spectra',
                                   X, [400, 1200],
                                   TA_spectra_norm,
                                   [0, 1.1*np.amax(TA_spectra_norm)], 'linear',
                                   ['Box 0', 'Box 1', 'Box 2', 'Box 3'])

graph3 = glob_ana_plot.kinetikos_2scale('Box filing kinetics',
                                        Y, [-1, 7000], 2,
                                        np.c_[laser_gauss(Y), model_kinetic],
                                        [0, 1.1*np.amax(model_kinetic)],
                                        ['Laser',
                                         'box 0',
                                         'box 1', 'box 2', 'box 3', 'box 4'])

graph4 = glob_ana_plot.kinetikos_2scale('Energy Averaged TA Kinetics',
                                        Y, [-1, 7000], 2,
                                        np.c_[TA_kinetic_experimental,
                                              TA_kinetic, TA_kinetic_sum, ],
                                        [0, 1.1*np.amax(TA_kinetic_sum)],
                                        ['Experimental ',
                                         'box 0', 'box 1', 'box 2', 'box 3',
                                         'Model sum'])

graph5 = glob_ana_plot.carpet_plot('Model',
                                   X, [450, 1000],
                                   Y, [-1, 7000], 2,
                                   rezultatas.best_fit,
                                   [-1, np.amax(rezultatas.best_fit),
                                    'tab20b'])

graph5 = glob_ana_plot.carpet_plot('Raw data',
                                   X, [450, 1000],
                                   Y, [0, 7000], 2,
                                   raw_carpet,
                                   [-1, np.amax(raw_carpet), 'tab20b'])

graph7 = glob_ana_plot.carpet_plot('Residual',
                                   X, [450, 1000],
                                   Y, [0, 7000], 2,
                                   raw_carpet-rezultatas.best_fit,
                                   [-1, 1, 'Spectral'])

# %% Išvedimas

# hehe, turėjo būti padaryta senai

# %% Pabaiga
import winsound

winsound.Beep(250, 500)
winsound.Beep(500, 500)
winsound.Beep(750, 500)

print('Elapsed time: {} s'.format(round(time.time()-t_start, 2)))
