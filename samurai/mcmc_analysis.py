import matplotlib as mpl
import platform
if platform.system() == "Linux":
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc

import numpy as np
import pdb
mpl.rc('font', family='Times New Roman')
mpl.rcParams['font.size'] = 25.0
from scipy.optimize import minimize
import sys, getopt
import corner
import datetime
import multiprocessing
import os
import h5py

# Specify directory of run to analyze
MCMC_DIR = ""

DIR = "mcmc_output/"

# Specify burn-in index for corner plot
DEFAULT_BURN_INDEX = 0

DEFAULT_WHICH = None

def estimate_burnin1(samples):
    # Determine time of burn-in by calculating first time median is crossed
    # Algorithm by Eric Agol 2016

    # Calculate the median for each parameter across all walkers and steps
    med_params = np.array([np.median(samples[:,:,i]) for i in range(nparam)])

    # Subtract off the median
    diff = samples - med_params

    # Determine where the sign changes occur
    asign = np.sign(diff)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

    # For each walker determine index where the first sign change occurs
    first_median_crossing = np.argmax(signchange>0, axis=1)

    # Now return the index of the last walker to cross its median
    return np.amax(first_median_crossing)

def median_crossing_burnin(samples):
    """
    Calculate the median crossing time for estimating the MCMC burn-in
    """

    # Unpack samples dimensions
    nwalk, niter, nparam = samples.shape

    for ip in range(nparam):
        for iw in range(nwalk):
            # Calculate chain medians
            med = np.median(samples[iw,:,ip])
            ii = 0
            # Subtract off the median
            diff = samples[iw,ii,ip] - med
            #isign =
            #while
            # Determine where the sign changes occur
            asign = np.sign(diff)
            signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)

        # For each walker determine index where the first sign change occurs
        first_median_crossing = np.argmax(signchange>0, axis=1)

def plot_trace(samples, directory="", names=None, which=None, alpha=0.2):

    print "Plotting Trace..."

    nwalkers = samples.shape[0]
    nsteps = samples.shape[1]
    nparam = samples.shape[2]

    """# Flatten chains (for histogram)
    print "Flattening chains for histogram (slow)..."
    samples_flat = samples[:, :, :].reshape((-1, nparam))
    """

    # Loop over all parameters making trace plots
    for i in range(nparam):
        if which is not None:
            i = which
        sys.stdout.write("\r{0}/{1}".format(i+1,nparam))
        sys.stdout.flush()
        if names is None:
            pname = ""
        else:
            if i > len(names) - 1:
                pname = r"Reg$_{%s}$" %(str(i-(len(names)-1)))
            else:
                pname = names[i]
        fig = plt.figure(figsize=(13,5))
        gs = gridspec.GridSpec(1,1)
        ax0 = plt.subplot(gs[0])
        ax0.plot(samples[:,:,i].T, lw=0.5, alpha=alpha)
        ax0.set_xlabel("Iteration")
        ax0.set_ylabel(pname+" Value")
        """# Add histogram on right
        ax1 = plt.subplot(gs[1], sharey=ax0)
        bins = np.linspace(np.min(samples_flat[:,i]), np.max(samples_flat[:,i]), 25, endpoint=True)
        h = ax1.hist(samples_flat[:,i], bins, orientation='horizontal', color="k", alpha=0.5)
        ax1.set_xticks([])
        ax1.yaxis.tick_right()
        ax1.yaxis.set_label_position("right")
        plt.setp(ax1.get_xticklabels(), fontsize=18, rotation=45)
        plt.setp(ax1.get_yticklabels(), fontsize=18, rotation=45)
        """
        plt.setp(ax0.get_xticklabels(), fontsize=18, rotation=45)
        plt.setp(ax0.get_yticklabels(), fontsize=18, rotation=45)
        fig.subplots_adjust(wspace=0)
        fig.savefig(os.path.join(directory, "trace"+str(i)+".png"), bbox_inches="tight")
        fig.clear()
        plt.close()
        if which is not None:
            break
    return

#===================================================

def run_mcmc_analysis(run, directory=DIR, iburn=DEFAULT_BURN_INDEX, which=DEFAULT_WHICH,
                      run_trace=False, run_corner=False):

    print "Burn-in index:", iburn

    MCMC_DIR = os.path.join(directory, run)

    # Load MCMC samples
    try:
        # Open the file stream
        hpath = os.path.join(MCMC_DIR, "samurai_out.hdf5")
        f = h5py.File(hpath, 'r')
    except IOError:
        print "Run directory does not exist! Check -d argument."
        sys.exit()

    # Extract info from HDF5 file
    samples = f["mcmc/samples"]
    N_TYPE = f.attrs["N_TYPE"]
    p0 = f["mcmc/p0"]
    X_names = f["mcmc"].attrs["X_names"]
    Y_names = f["mcmc"].attrs["Y_names"]
    nwalkers = samples.shape[0]
    nsteps = samples.shape[1]
    nparam = samples.shape[2]

    if run_trace:

        # Create directory for trace plots
        trace_dir = os.path.join(MCMC_DIR, "trace_plots/")
        try:
            os.mkdir(trace_dir)
            print "Created directory:", trace_dir
        except OSError:
            print trace_dir, "already exists."

        # Make trace plots
        plot_trace(samples, names=Y_names, directory=trace_dir, which=which)

    if run_corner:

        # Flatten chains
        print "Flattening chains..."
        samples_flat = samples[:,iburn:,:].reshape((-1, nparam))

        # Make corner plot
        print "Making Corner Plot..."
        fig = corner.corner(samples_flat, plot_datapoints=True, plot_contours=False, plot_density=False, labels=Y_names)
        fig.savefig(os.path.join(MCMC_DIR, "ycorner.png"))

    # Close HDF5 file stream
    f.close()

    # END

#===================================================

if __name__ == "__main__":

    # Read command line args
    myopts, args = getopt.getopt(sys.argv[1:],"d:b:w:")
    run = ""
    iburn = DEFAULT_BURN_INDEX
    which = DEFAULT_WHICH
    for o, a in myopts:
        # o == option
        # a == argument passed to the o
        if o == '-d':
            # Get MCMC directory timestamp name
            run=a
        elif o == "-b":
            # Get burn in index
            iburn = int(a)
        elif o == "-w":
            # Get which index
            which = int(a)
        else:
            pass

    #
    run_trace = False
    if "trace" in str(sys.argv):
        run_trace = True

    #
    run_corner = False
    if "corner" in str(sys.argv):
        run_corner = True

    run_mcmc_analysis(run, directory=DIR, iburn=iburn, which=which,
                      run_trace=run_trace, run_corner=run_corner)
