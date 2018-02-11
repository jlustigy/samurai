import matplotlib as mpl
import platform
if platform.system() == "Linux":
    mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc

import numpy as np
import healpy as hp
import emcee
from scipy.optimize import minimize
import sys, getopt
import corner
import datetime
import multiprocessing
import os
import h5py
import matplotlib.cm as cmx
import matplotlib.colors as colors
import pdb

from colorpy import colormodels, ciexyz

from reparameterize import transform_Y2X
from map_utils import save2hdf5

mpl.rc('font', family='Times New Roman')
mpl.rcParams['font.size'] = 25.0

DIR = "mcmc_output/"
EYECOLORS = False
DEFAULT_EPOXI = True

# Specify burn-in index for corner plot
DEFAULT_BURN_INDEX = 0

DEFAULT_WHICH = None

#---------------------------------------------------

def colorize(vector,cmap='plasma', vmin=None, vmax=None):
    """Convert a vector to RGBA colors.
    Parameters
    ----------
    vector : array
        Array of values to be represented by relative colors
    cmap : str (optional)
        Matplotlib Colormap name
    vmin : float (optional)
        Minimum value for color normalization. Defaults to np.min(vector)
    vmax : float (optional)
        Maximum value for color normalization. Defaults to np.max(vector)

    Returns
    -------
    vcolors : np.ndarray
        Array of RGBA colors
    scalarmap : matplotlib.cm.ScalarMappable
        ScalerMap to convert values to colors
    cNorm : matplotlib.colors.Normalize
        Color normalization
    """

    if vmin is None: vmin = np.min(vector)
    if vmax is None: vmax = np.max(vector)

    cm = plt.get_cmap(cmap)
    cNorm  = colors.Normalize(vmin=vmin, vmax=vmax)
    scalarmap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    vcolors = scalarmap.to_rgba(vector)

    return vcolors,scalarmap,cNorm

def decomposeX(x,n_band,n_slice,n_type):
    alb = x[0:n_band * n_type].reshape((n_type,n_band))
    area = x[n_band * n_type:].reshape((n_slice , n_type))
    return alb, area

def plot_median(med_alb, std_alb, med_area, std_area, n_all, directory="",
                epoxi=DEFAULT_EPOXI, eyecolors=False, lam=None):

    print "Plotting Median, Std..."

    fig = plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(1,2)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax0.set_ylabel("Area Fraction")
    ax0.set_xlabel("Slice #")
    ax1.set_ylabel("Albedo")

    xarea = np.arange(n_all["nslice"])
    xalb = np.arange(n_all["nband"])

    if n_all["nslice"] == n_all["ntimes"]:
        ax0.set_xlabel("Time [hrs]")
        ax0.set_xlim([np.min(xarea)-0.05, np.max(xarea)+0.05])
    else:
        ax0.set_xlabel("Slice Longitude [deg]")
        xarea = np.array([-180. + (360. / n_all["nslice"]) * (i + 0.5) for i in range(n_all["nslice"])])
        ax0.set_xlim([-185, 185])
        ax0.set_xticks([-180, -90, 0, 90, 180])

    if lam is not None:
        xalb = lam
        ax1.set_xlabel("Wavelength [nm]")
    elif epoxi:
        epoxi_bands = np.loadtxt("data/EPOXI_band")
        wl = epoxi_bands[:,1]
        xalb = wl
        ax1.set_xlabel("Wavelength [nm]")
        ax1.set_xlim([300,1000])
    else:
        ax1.set_xlabel("Band")
        ax1.set_xlim([np.min(xalb)-0.05, np.max(xalb)+0.05])

    if eyecolors:
        epoxi_bands = np.loadtxt("data/EPOXI_band")
        wl = epoxi_bands[:,1]
        c = [convolve_with_eye(wl, med_alb[i,:]) for i in range(n_all["ntype"])]
    else:
        c = ["purple", "orange", "green", "lightblue"]

    for i in range(n_all["ntype"]):
        ax0.plot(xarea, med_area[:,i], "o-", label="Surface %i" %(i+1), color=c[i])
        ax0.fill_between(xarea, med_area[:,i] - std_area[:,i], med_area[:,i] + std_area[:,i], alpha=0.3, color=c[i])
        ax1.plot(xalb, med_alb[i,:], "o-", color=c[i])
        ax1.fill_between(xalb, med_alb[i,:] - std_alb[i,:], med_alb[i,:] + std_alb[i,:], alpha=0.3 ,color=c[i])

    ax0.set_ylim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    leg=ax0.legend(loc=0, fontsize=14)
    leg.get_frame().set_alpha(0.0)

    fig.savefig(os.path.join(directory,"xmed_std.pdf"), bbox_inches="tight")

def plot_area_alb(samples, n_all, directory="", savetxt=True, intvls=[0.16, 0.5, 0.84],
                  epoxi=DEFAULT_EPOXI, eyecolors=False, lam=None):

    print "Plotting Area & Albedo..."

    nparam = samples.shape[1]

    # Compute grid of quantiles
    # q_l, q_50, q_h, q_m, q_p
    quantiles = np.array([nsig_intervals(samples[:,i], intvls=intvls)
        for i in range(nparam)])

    # Construct 2d arrays from 1d array
    alb_med, area_med = decomposeX(quantiles[:,1], n_all["nband"], n_all["nslice"], n_all["ntype"])
    alb_m, area_m = decomposeX(quantiles[:,3], n_all["nband"], n_all["nslice"], n_all["ntype"])
    alb_p, area_p = decomposeX(quantiles[:,4], n_all["nband"], n_all["nslice"], n_all["ntype"])

    # Make plot
    fig = plt.figure(figsize=(18,8))
    gs = gridspec.GridSpec(1,2)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax0.set_xlabel("Slice #")
    ax1.set_ylabel("Albedo")

    xarea = np.arange(n_all["nslice"])
    xalb = np.arange(n_all["nband"])

    if n_all["nslice"] == n_all["ntimes"]:
        ax0.set_xlabel("Time [hrs]")
        ax0.set_ylabel("Contribution Factor")
        ax0.set_xlim([np.min(xarea)-0.05, np.max(xarea)+0.05])
    else:
        ax0.set_xlabel("Slice Longitude [deg]")
        ax0.set_ylabel("Area Fraction")
        xarea = np.array([-180. + (360. / n_all["nslice"]) * (i + 0.5) for i in range(n_all["nslice"])])
        ax0.set_xlim([-185, 185])
        ax0.set_xticks([-180, -90, 0, 90, 180])

    if lam is not None:
        xalb = lam
        ax1.set_xlabel(r"Wavelength [$\mu$m]")
    elif epoxi:
        epoxi_bands = np.loadtxt("data/EPOXI_band")
        wl = epoxi_bands[:,1]
        xalb = wl
        ax1.set_xlabel("Wavelength [nm]")
        ax1.set_xlim([300,1000])
    else:
        ax1.set_xlabel("Band")
        ax1.set_xlim([np.min(xalb)-0.05, np.max(xalb)+0.05])

    if eyecolors:
        epoxi_bands = np.loadtxt("data/EPOXI_band")
        wl = epoxi_bands[:,1]
        c = [convolve_with_eye(wl, med_alb[i,:]) for i in range(n_all["ntype"])]
    else:
        # Set plot colors
        c = ["C%i" %(i%10) for i in range(n_all["ntype"])]

    for i in range(n_all["ntype"]):
        ax0.plot(xarea, area_med[:,i], "o-", label="Surface %i" %(i+1), color=c[i])
        ax0.fill_between(xarea, area_med[:,i] - area_m[:,i], area_med[:,i] + area_p[:,i], alpha=0.3, color=c[i])
        ax1.plot(xalb, alb_med[i,:], "o-", color=c[i], label="Surface %i" %(i+1))
        ax1.fill_between(xalb, alb_med[i,:] - alb_m[i,:], alb_med[i,:] + alb_p[i,:], alpha=0.3 ,color=c[i])

    ax0.set_ylim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    leg=ax1.legend(loc=0, fontsize=20)
    leg.get_frame().set_alpha(0.0)

    # Tweak tick labels
    plt.setp(ax0.get_xticklabels(), fontsize=18, rotation=0)
    plt.setp(ax0.get_yticklabels(), fontsize=18, rotation=0)
    plt.setp(ax1.get_xticklabels(), fontsize=18, rotation=0)
    plt.setp(ax1.get_yticklabels(), fontsize=18, rotation=0)

    # Save the plot
    fig.savefig(os.path.join(directory, "area-alb.pdf"), bbox_inches="tight")
    print "Saved:", "area-alb.pdf"

    return quantiles

def plot_sampling(x, directory="", epoxi=DEFAULT_EPOXI):

    ALPHA = 0.05

    print "Plotting %i Samples..." %len(x)

    fig = plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(1,2)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax0.set_ylabel("Area Fraction")
    ax0.set_xlabel("Slice #")
    ax1.set_ylabel("Albedo")

    xarea = np.arange(n_slice)
    xalb = np.arange(n_band)

    if n_slice == n_times:
        ax0.set_xlabel("Time [hrs]")
        ax0.set_xlim([np.min(xarea)-0.05, np.max(xarea)+0.05])
    else:
        ax0.set_xlabel("Slice Longitude [deg]")
        xarea = np.array([-180. + (360. / n_slice) * (i + 0.5) for i in range(n_slice)])
        ax0.set_xlim([-185, 185])
        ax0.set_xticks([-180, -90, 0, 90, 180])

    if epoxi:
        epoxi_bands = np.loadtxt("data/EPOXI_band")
        wl = epoxi_bands[:,1]
        xalb = wl
        ax1.set_xlabel("Wavelength [nm]")
        ax1.set_xlim([300,1000])
    else:
        ax1.set_xlabel("Band")
        ax1.set_xlim([np.min(xalb)-0.05, np.max(xalb)+0.05])

    if EYECOLORS:
        epoxi_bands = np.loadtxt("data/EPOXI_band")
        wl = epoxi_bands[:,1]
        c = [convolve_with_eye(wl, med_alb[i,:]) for i in range(N_TYPE)]
    else:
        c = ["purple", "orange", "green", "lightblue"]

    for s in range(len(x)):
        # Decompose x vector into albedo and area arrays
        alb, area = decomposeX(x[s], n_band, n_slice, N_TYPE)

        for i in range(N_TYPE):
            if s == 0:
                ax0.plot(0, 0, "-", label="Surface %i" %(i+1), color=c[i], alpha=1.0)
            ax0.plot(xarea, area[:,i], "-", color=c[i], alpha=ALPHA)
            ax1.plot(xalb, alb[i,:], "-", color=c[i], alpha=ALPHA)

    ax0.set_ylim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])

    leg=ax0.legend(loc=0, fontsize=16)
    leg.get_frame().set_alpha(0.0)

    fig.savefig(directory+"xsamples.pdf", bbox_inches="tight")

def convolve_with_eye(wl, spectrum):
    # Construct 2d array for ColorPy
    spec = np.vstack([wl, spectrum]).T
    # Call ColorPy modules to get irgb string
    rgb_eye = colormodels.irgb_string_from_rgb (
        colormodels.rgb_from_xyz (ciexyz.xyz_from_spectrum (spec)))
    return rgb_eye

def plot_reg1(samples):
    reg == 'Tikhonov'
    par = 'sigma', samples[-1]
    fig = plt.figure(figsize=(16,8))
    gs = gridspec.GridSpec(1,2)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax0.set_ylabel("Area Fraction")
    ax0.set_xlabel("Slice #")
    ax1.set_ylabel("Albedo")

def nsig_intervals(x, intvls=[0.16, 0.5, 0.84]):
    # Compute median and n-sigma intervals
    q_l, q_50, q_h = np.percentile(x, list(100.0 * np.array(intvls)))
    q_m, q_p = q_50-q_l, q_h-q_50
    return q_l, q_50, q_h, q_m, q_p

def plot_posteriors(samples, directory="", X_names=None, which=None, nbins=50):

    print "Plotting Posteriors..."

    nsteps = samples.shape[0]
    nparam = samples.shape[1]

    # Loop over all parameters making posterior histograms
    for i in range(nparam):
        if which is not None:
            i = which
        sys.stdout.write("\r{0}/{1}".format(i+1,nparam))
        sys.stdout.flush()
        if X_names is None:
            pname = ""
        else:
            pname = X_names[i]
        fig = plt.figure(figsize=(10,8))
        gs = gridspec.GridSpec(1,1)
        ax0 = plt.subplot(gs[0])
        ax0.set_xlabel(pname)
        ax0.set_ylabel("Relative Probability")
        bins = np.linspace(np.min(samples[:,i]), np.max(samples[:,i]), nbins, endpoint=True)
        h = ax0.hist(samples[:,i], bins, color="k", alpha=0.5)
        ax0.set_yticks([])
        plt.setp(ax0.get_xticklabels(), fontsize=18, rotation=45)
        plt.setp(ax0.get_yticklabels(), fontsize=18, rotation=45)
        q_l, q_50, q_h, q_m, q_p = nsig_intervals(samples[:,i], intvls=[0.16, 0.5, 0.84])
        # Format title
        title_fmt = ".3f"
        fmt = "{{0:{0}}}".format(title_fmt).format
        title = r"${{{0}}}_{{-{1}}}^{{+{2}}}$"
        title = title.format(fmt(q_50), fmt(q_m), fmt(q_p))
        ax0.set_title(title, y=1.01)
        # Plot vertical lines
        ax0.axvline(q_50, color="k", lw=2.0, ls="-")
        ax0.axvline(q_l, color="k", lw=2.0, ls="--")
        ax0.axvline(q_h, color="k", lw=2.0, ls="--")
        # Save
        fig.savefig(os.path.join(directory, "posterior"+str(i)+".png"), bbox_inches="tight")
        fig.clear()
        plt.close()
        if which is not None:
            break
    return

def plot_model_data(model_ij, Obs_ij, Obsnoise_ij, n_all, iburn=0, directory="",
                    show_all = True):

    print "Model-Data comparison..."

    nt = Obs_ij.shape[0]
    nb = Obs_ij.shape[1]

    ylabel = "Apparent Albedo"
    xlabel = "Time from start of observation [hrs]"

    time = np.arange(nt)+1.0
    colors = colorize(np.arange(nb), cmap="viridis")[0]
    labels = [r"$\lambda_{%i}$" %i for i in range(nb)]

    model = model_ij[iburn:,:,:,:].reshape(((-1, nt,nb)))

    intvls=[0.16, 0.5, 0.84]
    # Compute grid of quantiles
    # q_l, q_50, q_h, q_m, q_p

    quantiles = np.array([[nsig_intervals(model[:,i,j], intvls=intvls)
            for i in range(nt)] for j in range(nb)])

    medians = quantiles[:,:,1]
    qminus = quantiles[:,:,3]
    qplus = quantiles[:,:,4]

    for i in range(nb):
        # Make plot
        fig = plt.figure(figsize=(18,8))
        gs = gridspec.GridSpec(1,1)
        ax0 = plt.subplot(gs[0])
        ax0.set_ylabel(ylabel)
        ax0.set_xlabel(xlabel)
        ax0.plot(time, medians[i,:], alpha=1.0, color=colors[i], label=labels[i], lw=2.0)
        ax0.fill_between(time, medians[i,:] - qminus[i,:], medians[i,:] + qplus[i,:],
                    alpha=0.3, color=colors[i])
        ax0.errorbar(time, Obs_ij[:,i], yerr=Obsnoise_ij[:,i], c=colors[i],
                     fmt="o", ms=0, capsize=0, elinewidth=3)
        leg=ax0.legend(loc=0, fontsize=16)
        leg.get_frame().set_alpha(0.0)
        # Save
        fig.savefig(os.path.join(directory, "data_model"+str(i)+".png"), bbox_inches="tight")
        fig.clear()
        plt.close()

    # Make plot
    fig = plt.figure(figsize=(18,8))
    gs = gridspec.GridSpec(1,1)
    ax0 = plt.subplot(gs[0])
    ax0.set_ylabel(ylabel)
    ax0.set_xlabel(xlabel)
    for i in range(nb):
        ax0.plot(time, medians[i,:], alpha=1.0, color=colors[i], label=labels[i], lw=2.0)
        ax0.fill_between(time, medians[i,:] - qminus[i,:], medians[i,:] + qplus[i,:],
                    alpha=0.3, color=colors[i])
        ax0.errorbar(time, Obs_ij[:,i], yerr=Obsnoise_ij[:,i], c=colors[i],
                     fmt="o", ms=0, capsize=0, elinewidth=3)
    leg=ax0.legend(loc=0, fontsize=16)
    leg.get_frame().set_alpha(0.0)
    # Save
    fig.savefig(os.path.join(directory, "data_model_all"+str(i)+".png"), bbox_inches="tight")
    fig.clear()
    plt.close()

    return

#===================================================

def run_physical_mcmc_analysis(run, directory=DIR, run_sample=False, run_median=False, run_corner=False,
                           run_posterior=False, run_area_alb=False, run_model_data=False,
                           iburn=DEFAULT_BURN_INDEX,
                           which=DEFAULT_WHICH, eyecolors=False, epoxi=DEFAULT_EPOXI):

    print "Burn-in index:", iburn

    MCMC_DIR = os.path.join(directory, run)

    # Load MCMC samples
    try:
        # Open the file stream
        hpath = os.path.join(MCMC_DIR, "samurai_out.hdf5")
        f = h5py.File(hpath, 'r+')
    except IOError:
        print "Run directory does not exist! Check -d argument."
        sys.exit()

    # Extract info from HDF5 file
    samples = f["mcmc/samples"]
    model_ij = f["mcmc/model_ij"]
    N_TYPE = f.attrs["N_TYPE"]
    n_slice = f.attrs["N_SLICE"]
    p0 = f["mcmc/p0"]
    X_names = f["mcmc"].attrs["X_names"]
    Y_names = f["mcmc"].attrs["Y_names"]
    nwalkers = samples.shape[0]
    nsteps = samples.shape[1]
    nparam = samples.shape[2]
    # Unpack Data
    Obs_ij = f["data/Obs_ij"]
    Obsnoise_ij = f["data/Obsnoise_ij"]
    n_times = len(Obs_ij)
    n_band = len(Obs_ij[0])
    N_REGPARAM = f.attrs["N_REGPARAM"]

    # Put all the n's in a dictionary for easy access
    n_all = {
        "ntype" : N_TYPE,
        "nslice" : n_slice,
        "nwalkers" : nwalkers,
        "nsteps" : nsteps,
        "nparam" : nparam,
        "ntimes" : n_times,
        "nband" : n_band,
        "nregparam" : N_REGPARAM
    }

    # Throw assertion error if burn-in index exceeds number of steps
    assert iburn < samples.shape[1]

    # Compute slice longitude
    #slice_longitude = np.array([-180. + (360. / n_slice) * (i + 0.5) for i in range(n_slice)])

    NAME_XSAM = "physical_samples"

    # If the xsamples are already in the hdf5 file
    if NAME_XSAM in f["mcmc/"].keys():
        # load physical samples
        xs = f["mcmc/"+NAME_XSAM]
        print NAME_XSAM + " loaded from file!"
        if (xs.attrs["iburn"] == iburn) and (int(np.sum(xs[0,:])) != 0):
            # This is the exact same file or it has been loaded with 0's
            rerun = False
        else:
            # Must re-run xsamples with new burnin, overwrite
            print "Different burn-in index here. Must reflatten and convert..."
            rerun = True

    # If the xsamples are not in the hdf5 file,
    # or if they need to be re-run
    if NAME_XSAM not in f["mcmc/"].keys() or rerun:

        # Determine shape of new dataset
        nxparam = len(transform_Y2X(samples[0,0,:], N_TYPE, n_band, n_slice, flatten=True))
        new_shape = (nwalkers*(nsteps-iburn), nxparam)

        # Construct attrs dictionary
        adic = {"iburn" : iburn}

        # Delete existing dataset if it already exists
        if NAME_XSAM in f["mcmc/"].keys():
            del f["mcmc/"+NAME_XSAM]

        # Flatten chains
        print "Flattening chains beyond burn-in (slow, especially if low burn-in index)..."
        flat_samples = samples[:,iburn:,:].reshape((-1, nparam))

        # Different approach if there are regularization params
        if (N_REGPARAM > 0):
            print "Filling xsample dataset..."
            sys.stdout.flush()
            # Loop over walkers
            # Exclude regularization parameters from albedo, area samples
            xsam = np.array([transform_Y2X(flat_samples[i,:-1*N_REGPARAM],
                            N_TYPE, n_band, n_slice, flatten=True)
                            for i in range(len(flat_samples))]
                            )
        else:
            # Use all parameters
            xsam = np.array([transform_Y2X(flat_samples[i], N_TYPE, n_band,
                            n_slice, flatten=True)
                            for i in range(len(flat_samples))]
                            )

        # Create new dataset in existing hdf5 file
        xs = f.create_dataset("mcmc/"+NAME_XSAM, data=xsam, compression='lzf')
        # Add attributes to dataset
        for key, value in adic.iteritems(): xs.attrs[key] = value

    if run_sample:
        print "This functionality has been depreciated."
        """
        N_SAMP = 1000
        rand_sam = xs[np.random.randint(len(xs), size=N_SAMP),:]
        plot_sampling(rand_sam, directory=MCMC_DIR)
        """


    if run_median:

        print "Computing Median Parameters..."

        # Find median & standard deviation
        xmed = np.median(xs, axis=0)
        xstd = np.std(xs, axis=0)

        # Decompose into useful 2d arrays
        med_alb, med_area = decomposeX(xmed, n_band, n_slice, N_TYPE)
        std_alb, std_area = decomposeX(xstd, n_band, n_slice, N_TYPE)

        print "Median:", xmed
        print "Std:", xstd

        # Plot median
        plot_median(med_alb, std_alb, med_area, std_area, n_all, directory=MCMC_DIR,
                    epoxi=epoxi, eyecolors=eyecolors)

        # Save median results
        np.savetxt(os.path.join(MCMC_DIR, "albedo_median.txt"), np.vstack([med_alb, std_alb]).T)
        np.savetxt(os.path.join(MCMC_DIR, "area_median.txt"), np.vstack([med_area.T, std_area.T]).T)
        print "Saved:", "median_results.txt"

    if run_corner:

        print "Making Physical Corner Plot..."

        # Make corner plot
        fig = corner.corner(xs.value, plot_datapoints=False, plot_contours=True, plot_density=False,
            labels=X_names, show_titles=True)
        fig.savefig(os.path.join(MCMC_DIR, "xcorner.png"))

    if run_posterior:

        # Create directory for trace plots
        post_dir = os.path.join(MCMC_DIR, "physical_posteriors/")
        try:
            os.mkdir(post_dir)
            print "Created directory:", post_dir
        except OSError:
            print post_dir, "already exists."

        # Make posterior plots
        plot_posteriors(xs, X_names=X_names, directory=post_dir, which=which)

    if run_area_alb:

        # Define quantile intervals (1-sigma)
        intvls=[0.16, 0.5, 0.84]

        # Plot and save
        quantiles = plot_area_alb(xs, n_all, directory=MCMC_DIR, savetxt=True,
                                  intvls=intvls, epoxi=epoxi, eyecolors=eyecolors)

        # Delete save, if already exists
        if "quantiles" in f["mcmc/"].keys():
            del f["mcmc/quantiles"]
            print "Overwritting 'mcmc/quantiles' in hdf5 file"

        # Create new dataset
        qd = f["mcmc/"].create_dataset("quantiles", data=quantiles)

        # Create metadata dict
        dictionary = {
            "q_low" : 0,
            "q_50" : 1,
            "q_high" : 2,
            "q_minus" : 3,
            "q_plus" : 4,
            "intervals" : intvls
        }
        # Add metadata
        for key, value in dictionary.iteritems(): qd.attrs[key] = value

    if run_model_data:

        # Create directory
        plot_dir = os.path.join(MCMC_DIR, "model_data_compare/")
        try:
            os.mkdir(plot_dir)
            print "Created directory:", plot_dir
        except OSError:
            print plot_dir, "already exists."

        plot_model_data(model_ij, Obs_ij, Obsnoise_ij, n_all, iburn=iburn,
                        directory=plot_dir)

    # Close HDF5 file stream
    f.close()

    # END

#===================================================
if __name__ == "__main__":

    ###### Read command line args ######
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
    # Exit if no run directory provided
    if run == "":
        print("Please specify run directory using -d: \n e.g. >python mcmc_physical.py -d 2016-07-13--11-59")
        sys.exit()
    # Check for flag to convolve albedos with eye for plot colors
    if "eyecolors" in str(sys.argv):
        eyecolors = True
    else:
        eyecolors = False
    # Check for epoxi flag for wavelength labeling
    if "epoxi" in str(sys.argv):
        epoxi = True
    else:
        epoxi = False

    #
    run_sample = False
    if "sample" in str(sys.argv):
        run_sample = True

    #
    run_median=False
    if "median" in str(sys.argv):
        run_median=True

    #
    run_corner = False
    if "corner" in str(sys.argv):
        run_corner = True

    #
    run_posterior = False
    if "posterior" in str(sys.argv):
        run_posterior = True
    #
    run_area_alb = False
    if "area-alb" in str(sys.argv):
        run_area_alb = True

    #
    run_model_data = False
    if "model-data" in str(sys.argv):
        run_model_data = True

    # Call analysis function
    run_physical_mcmc_analysis(run, run_sample=run_sample, run_median=run_median,
                               run_corner=run_corner, run_posterior=run_posterior,
                               run_area_alb=run_area_alb, run_model_data=run_model_data, iburn=iburn,
                               which=which, eyecolors=eyecolors, epoxi=epoxi)

    ##################################
