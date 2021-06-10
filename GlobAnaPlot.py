# -*- coding: utf-8 -*-
"""
Created on Thu Jun 10 12:57:34 2021

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt

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
