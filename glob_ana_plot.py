# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:57:34 2021.

@author: Lenovo
"""

import matplotlib.pyplot as plt

CV_color_dict = {0: 'black', 1: 'red', 2: 'blue', 3: 'green', 4: 'purple',
                 5: 'orange'}


def carpet_plot(title,
                x_axis, x_lim,
                y_axis, y_lim, y_break,
                z_data, clr_min_max_mp):
    """
    Kilimo piešimo f-ja.

    Parameters
    ----------
    title : string, pavadinimas.
    x_axis : 1D np array, X ašis.
    x_lim: 2 element list, x_min, x_max.
    y_axis : 1D np array, Y ašis.
    y_lim: 2 element list, y_min, y_max.
    y_break: float, perėjimo iš lin į log vieta
    z_data : 2D np array, X*Y ašys dydžio.
    clr_min_max_mp : list, z_min, z_max, color map

    Returns
    -------
    Carpet plot Figure.

    """
    fig_c, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                     gridspec_kw={'height_ratios': [3, 1]})

    im1 = ax1.pcolormesh(x_axis, y_axis, z_data, shading='auto',
                         cmap=clr_min_max_mp[2])
    im1.set_clim(vmin=clr_min_max_mp[0], vmax=clr_min_max_mp[1])
    im2 = ax2.pcolormesh(x_axis, y_axis, z_data, shading='auto',
                         cmap=clr_min_max_mp[2])
    im2.set_clim(vmin=clr_min_max_mp[0], vmax=clr_min_max_mp[1])

    ax1.set_ylim(y_break, y_lim[1])
    ax1.set_yscale("log")
    ax2.set_ylim(y_lim[0], y_break)

    ax1.set_title(title)

    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.tick_params(labeltop=False)

    plt.xlim(x_lim[0], x_lim[1])

    cbar_ax = fig_c.add_axes([0.85, 0.15, 0.05, 0.6])
    fig_c.colorbar(im2, ax=ax1, cax=cbar_ax)
    fig_c.subplots_adjust(hspace=0)
    fig_c.subplots_adjust(right=0.825)

    fig_c.text(0.04, 0.5, 'Delay (ps)', va='center', ha='center',
               rotation='vertical', fontsize=12)
    fig_c.text(0.5, 0.03, 'Probe energy (eV)', va='center', ha='center',
               fontsize=12)
    fig_c.text(0.85, 0.78, r'   $\Delta$T'+' \n(mOD)', fontsize=12)

    return fig_c


def kinetikos_2scale(title,
                     x_scale, x_lim, x_break,
                     y_data, y_lim,
                     legend):
    """
    Kinetikų su dviem skalem plotinimo f-ja.

    Parameters
    ----------
    title : string, pavadinimas.
    x_scale : 1D np array, X ašis.
    x_lim :  2 element list, x_min, x_max.
    x_break : float, perėjimo iš lin į log vieta.
    y_data : np array, stulpeliai plotinimui.
    y_lim :  2 element list, y_min, y_max.
    legend : list of strings, grafikų pavadinimai.

    Returns
    -------
    kinetic plot Figure.
    """
    fig_k, (ax1, ax2) = plt.subplots(1, 2, sharey=True,
                                     gridspec_kw={'width_ratios': [1, 3]})

    title = title

    for idx in range(len(y_data[0, :])):
        # print(len(box_kin[0, :]), idx)
        ax1.plot(x_scale, y_data[:, idx],
                 color=CV_color_dict[idx])
        ax2.plot(x_scale, y_data[:, idx],
                 label=legend[idx], color=CV_color_dict[idx])

    ax1.set_xlim(x_lim[0], x_break)
    ax2.set_xlim(x_break, x_lim[1])
    ax2.set_xscale("log")
    ax1.set_ylim(y_lim[0], y_lim[1])

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

    return fig_k


def TA_spectral(title,
                x_scale, x_lim,
                y_data, y_lim, y_scale,
                legend):
    """
    TA spekro piešimo f-ja.

    Parameters
    ----------
    title : string, pavadinimas.
    x_scale : 1D np array, X ašis.
    x_lim :  2 element list, x_min, x_max.
    y_data : np array, stulpeliai plotinimui.
    y_lim :  2 element list, y_min, y_max.
    legend : list of strings, grafikų pavadinimai.

    Returns
    -------
    Spectra plot Figure.
    """
    fig_t = plt.figure()
    ax0 = fig_t.add_subplot(1, 1, 1)

    fig_t.text(0.5, 0.85, title, va='center', ha='center', fontsize=12)

    for idx in range(len(y_data[0, :])):
        ax0.plot(x_scale, y_data[:, idx], color=CV_color_dict[idx+1],
                 label=legend[idx])

    ax0.set_xlim(x_lim[0], x_lim[1])
    ax0.set_ylim(y_lim[0], y_lim[1])
    ax0.set_yscale(y_scale)

    fig_t.subplots_adjust(top=0.80)
    fig_t.text(0.04, 0.5, 'TA signal (a.u.)', va='center', ha='center',
               rotation='vertical', fontsize=12)
    fig_t.text(0.5, 0.03, 'Probe energy (eV)', va='center', ha='center',
               fontsize=12)
    fig_t.legend()

    return fig_t


