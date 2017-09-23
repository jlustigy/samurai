# Import the Surface Albedo Mapping Using RotAtional Invsersion (SAMURAI) model
import samurai
import h5py
import numpy as np

dpath = "exm_cg1.hdf5"

# Set some run parameters
fmodel = "lightcurve"
imodel = "emcee3"
Nmcmc = 50000
ntype = 3
reg = None
nslice = 13
mcmc_seedamp = 1.5

# Define a dataset to analyze
# Open the file stream
f = h5py.File(dpath, 'r')
Obs = f["obs"].value
Obsnoise = f["sig"].value
wlc = f["lam"].value
wlw = f["dlam"].value
tgrid = np.arange(wlw.shape[0])

data = samurai.Data(Time_i=tgrid, Obs_ij=Obs, Obsnoise_ij=Obsnoise, wlc_i=wlc, wlw_i=wlw)
dtag = "exm1"

tag = dtag+"_"+fmodel+"_"+imodel+"_1.5mseed"

# Define a Mapper object
sim = samurai.Mapper(data=data, fmodel=fmodel, imodel=imodel, Nmcmc=Nmcmc, ntype=ntype,
                     regularization=reg, mcmc_seedamp=mcmc_seedamp, nslice=nslice)

# Run samurai simulation
sim.run_mcmc(tag=tag, savedir=u'/Users/jlustigy/Documents/')

# Open output from sim
sim.output.open()

# Plot mcmc chain trace plots
sim.output.plot_trace()

# Close output file
sim.output.close()
