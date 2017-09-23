# Import the Surface Albedo Mapping Using RotAtional Invsersion (SAMURAI) model
import samurai

# Define path to hdf5 output to load from
#hpath = "mcmc_output/simpleIGBP_emcee3_GP/samurai_out.hdf5"
hpath = "mcmc_output/simpleIGBP_lightcurve_emcee3_GP/samurai_out.hdf5"
Nmcmc = 50000

# Load Mapper object from hdf5 file
sim = samurai.Mapper.from_hdf5(hpath)

# Set new number of iterations
sim.Nmcmc = Nmcmc

# Run samurai simulation
sim.run_mcmc(resume=True)

# Open output from sim
sim.output.open()

# Plot mcmc chain trace plots
sim.output.plot_trace()

# Close output file
sim.output.close()
