#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''
run_samurai_EPOXI.py
--------------------

Run `samurai.Mapper` on the EPOXI observations of Earth
'''

from __future__ import division, print_function, absolute_import, unicode_literals

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib as mpl
import os, sys
sys.path.insert(1, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import samurai as sam

mpl.rc('font',**{'family':'serif','serif':['Computer Modern']})
mpl.rcParams['font.size'] = 25.0
mpl.rc('text', usetex=True)

data = sam.Data.from_EPOXI_june()

sim = sam.Mapper(data = data, ntype = 4, fmodel="map",
                 nslice=13, use_global = False, use_grey = True)

tag = "EPOXI_June_map1"
sim.run_oe_atmosphere(N=100, savedir="output", tag=tag, verbose=True)

run_path = os.path.split(sim.output.hpath)[0]

hpath = sim.output.hpath

f = h5py.File(hpath, "r")
Noe = len(f['oe'].keys())

fig, ax  = plt.subplots(1, 2,figsize=(16,6))

n_type = f.attrs["ntype"]
Obs_ij = f["data/Obs_ij"]
n_times = len(Obs_ij)
n_band = len(Obs_ij[0])
n_regparam = f.attrs["nregparam"]
time = np.arange(n_times)
wl = f['data/wlc_i'].value
dwl = f['data/wlw_i'].value

xalb = wl
ax[1].set_xlabel("Wavelength [$\mu$m]")
ax[1].set_ylabel("Albedo")

# Set n_slice based on forward model
if f.attrs["fmodel"] == "map":
    n_slice = f.attrs["nslice"]
    ax[0].set_xlabel("Slice Longitude [deg]")
    ax[0].set_ylabel("Area Fraction")
    xarea = np.array([-180. + (360. / n_slice) * (i + 0.5) for i in range(n_slice)])
    ax[0].set_xlim([-185, 185])
    ax[0].set_xticks([-180, -90, 0, 90, 180])
elif f.attrs["fmodel"] == "lightcurve":
    n_slice = len(Obs_ij)
    ax[0].set_xlabel("Time [hrs]")
    ax[0].set_ylabel("Contribution Factor")
    xarea = time
else:
    print("Error: %s is an unrecognized forward model" %f.attrs["fmodel"])

lnps = np.array([f['oe/%i/' %i].attrs['best_lnprob'] for i in range(Noe)])
ibest = np.argmin(lnps)

i = ibest

print("BIC : %.3f" %f['oe/%i/' %i].attrs['best_BIC'])
print("lnProb %.3f:" %f['oe/%i/' %i].attrs['best_lnprob'])

X_area_lk = f['oe/%i/X_area_lk' %i].value
X_albd_kj_T = f['oe/%i/X_albd_kj_T' %i].value

for j in range(n_type):
    ax[0].plot(xarea, X_area_lk[:,j], "o-", label="Surface %i" %(j+1), color="C%i" %j, alpha=1.0)
    ax[1].plot(xalb, X_albd_kj_T[:,j], "o-", color="C%i" %j, alpha=1.0, label="Surface %i" %(j+1))

# Make legend
leg=ax[1].legend(loc=0, fontsize=20)
leg.get_frame().set_alpha(0.0)

# Tweak tick labels
plt.setp(ax[0].get_xticklabels(), fontsize=18, rotation=0)
plt.setp(ax[0].get_yticklabels(), fontsize=18, rotation=0)
plt.setp(ax[1].get_xticklabels(), fontsize=18, rotation=0)
plt.setp(ax[1].get_yticklabels(), fontsize=18, rotation=0)

fig.savefig("../plots/"+tag+".pdf", bbox_inches="tight")
