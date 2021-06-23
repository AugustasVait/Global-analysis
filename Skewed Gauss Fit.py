# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 13:03:17 2021.

@author: AV

Skeweg Gauss fitting

Duomenys kaip visada paimami iš carpet view pirminio modeliavimo.
"""

# %% import modules and data.

import numpy as np
import lmfit
import glob_ana_plot


spektrai = np.loadtxt("C:\\VU FNI Cloud\\CarpetView Modeling\\GYAGG CV" +
                      "\\4box models\\All F\\Final for paper\\" +
                      "\\pp_g68_340_1,5uJ_30kHz_eV.dat_amplitudes.dat")

# %% pradinis raw duomenu apdorojimas

BoxNr = 3  # kuris spektras tiriamas

ev_axis = spektrai[1:, 0]
box_spectra = spektrai[1:, BoxNr+1]

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


# %% Modeliavimas

modelis = lmfit.Model(skewed_g, independent_vars=['X'])

# print('parameter names: {}'.format(modelis.param_names))

modelis.set_param_hint('A', value=20, min=0, max=1e5, vary=True)
modelis.set_param_hint('x0', value=1.5, min=1, max=3, vary=True)
modelis.set_param_hint('W', value=1, min=0, max=3, vary=True)
modelis.set_param_hint('b', value=0, min=-2, max=2, vary=True)
modelis.set_param_hint('Y0', value=1.0, min=0, max=1e5, vary=True)

parametrai = modelis.make_params()

rezultatas = modelis.fit(box_spectra, X=ev_axis, params=parametrai,
                         method='leastsq')

print(lmfit.fit_report(rezultatas))

# %% Vaizdinimas

glob_ana_plot.TA_spectral('title',
                          ev_axis, [1.3, 2.6],
                          np.c_[box_spectra, rezultatas.best_fit],
                          [0.0, 1.1*np.amax(box_spectra)],
                          ['Box spectra', 'Model'])
