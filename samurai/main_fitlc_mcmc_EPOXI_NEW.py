import numpy as np
import healpy as hp
import emcee
from scipy.optimize import minimize
import sys
import datetime
import multiprocessing
import os
from pdb import set_trace as stop
import h5py

__all__ = ["run_lightcurve_mcmc"]

from fitlc_params import NUM_MCMC, NUM_MCMC_BURNIN, SEED_AMP, SIGMA_Y, NOISELEVEL, \
    REGULARIZATION, N_TYPE, deg2rad, N_SIDE, INFILE, calculate_walkers, HDF5_COMPRESSION, \
    WAVEBAND_CENTERS, WAVEBAND_WIDTHS

import prior
import reparameterize
from map_utils import generate_tex_names, save2hdf5

NCPU = multiprocessing.cpu_count()


# March 2008
#LAT_S = -0.5857506  # sub-solar latitude
#LON_S = 267.6066184  # sub-solar longitude
#LAT_O = 1.6808370  # sub-observer longitude
#LON_O = 210.1242232 # sub-observer longitude


#===================================================
# basic functions
#=============================================== ====

N_REGPARAM = 0
if REGULARIZATION is not None:
    if REGULARIZATION == 'Tikhonov' :
        N_REGPARAM = 1
    elif REGULARIZATION == 'GP' :
        N_REGPARAM = 3
    elif REGULARIZATION == 'GP2' :
        N_REGPARAM = 2
else :
    N_REGPARAM = 0

#---------------------------------------------------
def lnprob(Y_array, *args):
    """
    Misfit-function to be minimized
    """

    # Unpack args
    Obs_ij, Obsnoise_ij, Kernel_il, N_REGPARAM, flip, verbose  = args
    n_slice = len(Obs_ij)
    n_band = len(Obs_ij[0])

    # Parameter conversion
    if (N_REGPARAM > 0):
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X(Y_array[:-1*N_REGPARAM], N_TYPE, n_band, n_slice )
    else:
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X(Y_array, N_TYPE, n_band, n_slice )

    # Model
    Model_ij = np.dot(Kernel_il, np.dot(X_area_lk, X_albd_kj))

    # Chi-squared statistic
    Diff_ij = ( Obs_ij - Model_ij ) / Obsnoise_ij
    Chi2_i  = np.diag(np.dot( Diff_ij, Diff_ij.T ))
    chi2    = np.sum(Chi2_i)

    # Flat prior for albedo
    Y_albd_kj = Y_array[0:N_TYPE*n_band].reshape([N_TYPE, n_band])
    ln_prior_albd = prior.get_ln_prior_albd( Y_albd_kj )

    # flat prior for area fraction
    Y_area_lk = Y_array[N_TYPE*n_band:N_TYPE*n_band+n_slice*(N_TYPE-1)].reshape([n_slice, N_TYPE-1])
    ln_prior_area = prior.get_ln_prior_area_new( Y_area_lk, X_area_lk[:,:-1] )

    # flat ordering prior for labeling degeneracy
    ln_prior_order = prior.get_ln_prior_ordering(X_albd_kj, X_area_lk)

    # regularization
    # ---Tikhonov Regularization
    if REGULARIZATION is not None:
        if ( REGULARIZATION == 'Tikhonov' ):
            regparam = Y_array[-1*N_REGPARAM]
            regterm_area = prior.regularize_area_tikhonov( X_area_lk, regparam )
    # ---Gaussian Process
        elif ( REGULARIZATION == 'GP' ):
            regparam = ( Y_array[-1*N_REGPARAM], Y_array[-1*N_REGPARAM+1], Y_array[-1*N_REGPARAM+2] )
            regterm_area = prior.regularize_area_GP( X_area_lk, regparam )
    # ---Gaussian Process without constraint
        elif ( REGULARIZATION == 'GP2' ):
            regparam = ( Y_array[-1*N_REGPARAM], Y_array[-1*N_REGPARAM+1] )
            regterm_area = prior.regularize_area_GP2( X_area_lk, regparam )
    # ---Others
    else :
        regterm_area = 0.

    # verbose
    if verbose :
        print 'chi2', chi2 - ln_prior_albd - ln_prior_area, chi2, ln_prior_albd, ln_prior_area
        print 'chi2/d.o.f.', chi2 / (len(Y_array)*1.-1.), len(Y_array)

    answer = - chi2 + ln_prior_albd + ln_prior_area + ln_prior_order + regterm_area

    # Check for nans
    if np.isnan(answer):
        answer = -np.inf

    if flip :
        return -1. * answer
    else :
         return answer, Model_ij

#---------------------------------------------------
def run_initial_optimization(lnlike, data, guess, method="Nelder-Mead", run_dir=""):

    print "Finding initial best-fit values using %s method..." %method

    # Decompose data
    Obs_ij = data[0]
    n_slice = len(Obs_ij)
    n_band = len(Obs_ij[0])
    n_regparam = data[3]

    # Run optimization
    output = minimize(lnlike, guess, args=data, method=method)

    # Get best-fitting params
    best_fit = output["x"]
    print "initial best-fit:", best_fit

    # Get best-lnlike and BIC
    lnprob_bestfit = lnlike( output['x'], *data )
    BIC = 2.0 * lnprob_bestfit + len( output['x'] ) * np.log( len(Obs_ij.flatten()) )
    print 'BIC: ', BIC

    # Transform back to physical params
    if (n_regparam > 0):
        X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(output["x"][:-1*n_regparam], N_TYPE, n_band, n_slice )
    else:
        X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(output["x"], N_TYPE, n_band, n_slice )
    #X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(output["x"], N_TYPE, n_band, n_slice )
    X_albd_kj_T = X_albd_kj.T

    # Flatten best-fitting physical parameters
    bestfit = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

    # Calculate residuals
    residuals = Obs_ij - np.dot( X_area_lk, X_albd_kj )
    #print "residuals", residuals

    # Create dictionaries of initial results to convert to hdf5
    # datasets and attributes
    init_dict_datasets = {
        "best_fity" : best_fit,
        "X_area_lk" : X_area_lk,
        "X_albd_kj_T" : X_albd_kj_T,
        "best_fitx" : bestfit
    }
    init_dict_attrs = {
        "best_lnprob" : lnprob_bestfit,
        "best_BIC" : BIC
    }

    """
    # Save initialization run as npz
    print "Saving:", run_dir+"initial_minimize.npz"
    np.savez(run_dir+"initial_minimize.npz", data=data, best_fity=best_fit, \
        lnprob_bestfit=lnprob_bestfit, BIC=BIC, X_area_lk=X_area_lk, \
        X_albd_kj_T=X_albd_kj_T, residuals=residuals, best_fitx =bestfit)
    """

    return (init_dict_datasets, init_dict_attrs)

#===================================================
#if __name__ == "__main__":
def run_lightcurve_mcmc():
    """
    """

    # print start time
    now = datetime.datetime.now()
    print now.strftime("%Y-%m-%d %H:%M:%S")

    # Create directory for this run
    startstr = now.strftime("%Y-%m-%d--%H-%M")
    run_dir = "mcmc_output/" + startstr + "/"
    os.mkdir(run_dir)
    print "Created directory:", run_dir

    # Save THIS file and the param file for reproducibility!
    thisfile = os.path.basename(__file__)
    paramfile = "fitlc_params.py"
    newfile = os.path.join(run_dir, thisfile)
    commandString1 = "cp " + thisfile + " " + newfile
    commandString2 = "cp "+paramfile+" " + os.path.join(run_dir,paramfile)
    os.system(commandString1)
    os.system(commandString2)
    print "Saved :", thisfile, " &", paramfile

    # input data
    Obs_ij = np.loadtxt(INFILE)
    n_slice = len(Obs_ij)

    n_band = len(Obs_ij[0])
    Time_i = np.arange( n_slice )

    Obsnoise_ij = ( NOISELEVEL * Obs_ij )

    # set kernel
#    Kernel_il = kernel(Time_i, n_slice)
    Kernel_il = np.identity( n_slice )
#    Sigma_ll = np.identity(n_slice)

#    print 1/0
#    set initial condition
#    Y0_array = np.ones(N_TYPE*n_band+n_slice*(N_TYPE-1))
    X0_albd_kj = 0.3+np.zeros([N_TYPE, n_band])
    X0_area_lk = 0.1+np.zeros([n_slice, N_TYPE])
    """ # Load perfect starting position from file
    temp = np.load("mockdata/mock_simple_3types_1_albd_area.npz")
    X0_albd_kj = temp["X0_albd_kj"]
    X0_area_lk = temp["X0_area_lk"]
    """

    # Create list of strings for Y & X parameter names
    Y_names, X_names = generate_tex_names(N_TYPE, n_band, n_slice)

    Y0_array = reparameterize.transform_X2Y(X0_albd_kj, X0_area_lk)

    if ( N_REGPARAM > 0 ) :
        Y0_array = np.append(Y0_array, np.array([10.]*N_REGPARAM) )

    n_dim = len(Y0_array)
    print '# of parameters', n_dim

#    Y0_albd_kj = np.zeros([N_TYPE,  len(Obs_ij[0])])
#    Y0_area_lk = np.zeros([n_slice, N_TYPE-1])
#    Y0_area_lk[:,0] = 1.
#    Y0_list = [Y0_albd_kj, Y0_area_lk]
#    print "Y0_array", Y0_array

    if (N_REGPARAM > 0):
        X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(Y0_array[:-1*N_REGPARAM], N_TYPE, n_band, n_slice )
    else:
        X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(Y0_array, N_TYPE, n_band, n_slice )

#    print "X_area_lk", X_area_lk
#    print "X_albd_kj", X_albd_kj

    ########## use optimization for mcmc initial guesses ##########

    data = (Obs_ij, Obsnoise_ij, Kernel_il, N_REGPARAM, True, False)

    init_dict_datasets, init_dict_attrs = run_initial_optimization(lnprob, data, Y0_array, method="Nelder-Mead", run_dir=run_dir)
    best_fit = init_dict_datasets["best_fity"]

    ########## Run MCMC ##########

    # Number of dimensions is number of free parameters
    n_dim = len(Y0_array)
    # Number of walkers
    n_walkers = calculate_walkers(n_dim)

    # Data tuple to pass to emcee
    data = (Obs_ij, Obsnoise_ij, Kernel_il, N_REGPARAM, False, False)

    # Initialize emcee EnsembleSampler object
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, args=data, threads=NCPU)

    # Set starting guesses as gaussian noise ontop of intial optimized solution
    # note: consider using emcee.utils.sample_ball(p0, std) (std: axis-aligned standard deviation.)
    #       to produce a ball of walkers around an initial parameter value.
    p0 = SEED_AMP*np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim)) + best_fit

    if NUM_MCMC_BURNIN > 0:
        print "MCMC until burn-in..."
        # Run MCMC
        pos, prob, state = sampler.run_mcmc( p0, NUM_MCMC_BURNIN )
        # Save initial positions of chain[n_walkers, steps, n_dim]
        burnin_chain = sampler.chain[:, :, :].reshape((-1, n_dim))
        # Save chain[n_walkers, steps, n_dim] as npz
        now = datetime.datetime.now()
        print "Finished Burn-in MCMC:", now.strftime("%Y-%m-%d %H:%M:%S")
        print "Saving:", run_dir+"mcmc_burnin.npz"
        np.savez(run_dir+"mcmc_burnin.npz", pos=pos, prob=prob, burnin_chain=burnin_chain)
        print "MCMC from burn-in..."
        # Set initial starting position to the current state of chain
        p0 = pos
        # Reset sampler for production run
        sampler.reset()
    else:
        print "MCMC from initial optimization..."

    # Run MCMC
    sampler.run_mcmc( p0, NUM_MCMC )

    # Get emcee chain samples
    original_samples = sampler.chain

    # Get model evaluations
    blobs = sampler.blobs
    shape = (len(blobs), len(blobs[0]), len(blobs[0][0]), len(blobs[0][0][0]))
    model_ij = np.reshape(blobs, shape)

    ############ Save HDF5 File ############

    # Specify hdf5 save file and group names
    hfile = os.path.join(run_dir, "samurai_out.hdf5")
    grp_init_name = "initial_optimization"
    grp_mcmc_name = "mcmc"
    grp_data_name = "data"
    compression = HDF5_COMPRESSION

    # print
    print "Saving:", hfile

    # dictionary for global run metadata
    hfile_attrs = {
        "N_TYPE" : N_TYPE,
        "N_SLICE" : n_slice,
        "N_REGPARAM" : N_REGPARAM
    }

    # Create dictionaries for mcmc data and metadata
    mcmc_dict_datasets = {
        "samples" : original_samples,
        "model_ij" : model_ij,
        "p0" : p0
    }
    mcmc_dict_attrs = {
        "Y_names" : Y_names,
        "X_names" : X_names,
    }

    # Create dictionaries for observation data and metadata
    data_dict_datasets = {
        "Obs_ij" : Obs_ij,
        "Obsnoise_ij" : Obsnoise_ij,
        "Kernel_il" : Kernel_il,
        "lam_j" : WAVEBAND_CENTERS,
        "dlam_j" : WAVEBAND_WIDTHS
    }
    data_dict_attrs = {
        "datafile" : INFILE
    }

    # Create hdf5 file
    f = h5py.File(hfile, 'w')

    # Add global metadata
    for key, value in hfile_attrs.iteritems(): f.attrs[key] = value

    # Create hdf5 groups (like a directory structure)
    grp_init = f.create_group(grp_init_name)    # f["initial_optimization/"]
    grp_data = f.create_group(grp_data_name)    # f["data/"]
    grp_mcmc = f.create_group(grp_mcmc_name)    # f[mcmc/]

    # Save initial run datasets
    for key, value in init_dict_datasets.iteritems():
        grp_init.create_dataset(key, data=value, compression=compression)
    # Save initial run metadata
    for key, value in init_dict_attrs.iteritems():
        grp_init.attrs[key] = value

    # Save data datasets
    for key, value in data_dict_datasets.iteritems():
        grp_data.create_dataset(key, data=value, compression=compression)
    # Save data metadata
    for key, value in data_dict_attrs.iteritems():
        grp_data.attrs[key] = value

    # Save mcmc run datasets
    for key, value in mcmc_dict_datasets.iteritems():
        grp_mcmc.create_dataset(key, data=value, compression=compression)
    # Save mcmc run metadata
    for key, value in mcmc_dict_attrs.iteritems():
        grp_mcmc.attrs[key] = value

    # Close hdf5 file stream
    f.close()
