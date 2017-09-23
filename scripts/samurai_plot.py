# Import the Surface Albedo Mapping Using RotAtional Invsersion (SAMURAI) model
import samurai

# Define path to hdf5 output to load from
#hpath = "mcmc_output/simpleIGBP_emcee3_GP/samurai_out.hdf5"
hpath = "/Users/jlustigy/Documents/exm1_lightcurve_emcee3_1.5mseed/samurai_out.hdf5"

# set burn-in index
iburn = 40000

# Load Mapper object from hdf5 file
sim = samurai.Mapper.from_hdf5(hpath)

# Open output from sim
sim.output.open()

# 
sim.output.transform_samples(iburn)

#
sim.output.plot_posteriors()

#
sim.output.plot_area_alb()

# Close output file
sim.output.close()
