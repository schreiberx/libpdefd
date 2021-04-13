#! /usr/bin/env python

import libpdefd.plot_config as pc
import atmos_consts as ac
import matplotlib.pyplot as plt



"""
Temperature
"""

for xscale in ['linear', 'log']:

    fig, ax = pc.setup(ncols=2, nrows=2)
    ps = pc.PlotStyles()


    ax[0,0].plot(ac.density, ac.altitude*1e-3)
    ax[0,0].set_ylabel("Height $h$ in $km$")
    ax[0,0].set_xlabel("Density")
    ax[0,0].set_title("Density profile")
    ax[0,0].set_xscale(xscale)


    ax[0,1].plot(ac.pressure, ac.altitude*1e-3)
    ax[0,1].set_ylabel("Height in $km$")
    ax[0,1].set_xlabel("Pressure")
    ax[0,1].set_title("Pressure profile")
    ax[0,1].set_xscale(xscale)


    ax[1,0].plot(ac.temperature, ac.altitude*1e-3)
    ax[1,0].set_ylabel("Height in $km$")
    ax[1,0].set_xlabel("Temperature")
    ax[1,0].set_title("Temperature profile")
    ax[1,0].set_xscale(xscale)


    R = 287
    temp_reconstructed = ac.pressure/(ac.density*R)
    ax[1,1].plot(temp_reconstructed, ac.altitude*1e-3)
    ax[1,1].set_ylabel("Height in $km$")
    ax[1,1].set_xlabel("Temperature (reconstructed)")
    ax[1,1].set_title("Temperature profile (reconstructed)")
    ax[1,1].set_xscale(xscale)


    fig.tight_layout()
    filename="output_atmos_consts_temperature_profile_"+xscale+".pdf"

    print("Filename: "+filename)
    fig.savefig(filename)

