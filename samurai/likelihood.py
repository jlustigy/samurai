import numpy as np

import prior
import reparameterize

__all__ = ["lnprob", "lnprob_atmosphere"]

################################################################################

def lnprob(Y_array, *args):
    """
    Log-probability function for mapping

    Parameters
    ----------

    Returns
    -------
    """

    # Unpack args
    Obs_ij, Obsnoise_ij, Kernel_il, regularization, n_regparam, flip, verbose, n_type, n_slice  = args
    n_band = len(Obs_ij[0])

    # Parameter conversion
    if (n_regparam > 0):
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X(Y_array[:-1*n_regparam], n_type, n_band, n_slice )
    else:
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X(Y_array, n_type, n_band, n_slice )

    # Model
    Model_ij = np.dot(Kernel_il, np.dot(X_area_lk, X_albd_kj))

    # Chi-squared statistic
    Diff_ij = ( Obs_ij - Model_ij ) / Obsnoise_ij
    Chi2_i  = np.diag(np.dot( Diff_ij, Diff_ij.T ))
    chi2    = np.sum(Chi2_i)

    # Flat prior for albedo
    Y_albd_kj = Y_array[0:n_type*n_band].reshape([n_type, n_band])
    ln_prior_albd = prior.get_ln_prior_albd( Y_albd_kj )

    # flat prior for area fraction
    Y_area_lk = Y_array[n_type*n_band:n_type*n_band+n_slice*(n_type-1)].reshape([n_slice, n_type-1])
    ln_prior_area = prior.get_ln_prior_area_new( Y_area_lk, X_area_lk[:,:-1] )

    # flat ordering prior for labeling degeneracy
    ln_prior_order = prior.get_ln_prior_ordering(X_albd_kj, X_area_lk)

    # regularization
    if regularization is not None:
        if ( regularization == 'Tikhonov' ):
            # ---Tikhonov Regularization
            regparam = Y_array[-1*n_regparam]
            regterm_area = prior.regularize_area_tikhonov( X_area_lk, regparam )
        elif ( regularization == 'GP' ):
            # ---Gaussian Process
            regparam = ( Y_array[-1*n_regparam], Y_array[-1*n_regparam+1], Y_array[-1*n_regparam+2] )
            regterm_area = prior.regularize_area_GP( X_area_lk, regparam )
        elif ( regularization == 'GP2' ):
            # ---Gaussian Process without constraint
            regparam = ( Y_array[-1*n_regparam], Y_array[-1*n_regparam+1] )
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

def lnprob_atmosphere(Y_array, *args):
    """
    Log-probability function for mapping

    Parameters
    ----------

    Returns
    -------
    """

    # Unpack args
    Obs_ij, Obsnoise_ij, Kernel_il, regularization, n_regparam, flip, verbose, n_type, n_slice, use_grey, use_global  = args
    n_band = len(Obs_ij[0])

    # Parameter conversion
    if (n_regparam > 0):
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X_atmosphere(Y_array[:-1*n_regparam], n_type, n_band, n_slice, use_grey=use_grey, use_global=use_global )
    else:
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X_atmosphere(Y_array, n_type, n_band, n_slice, use_grey=use_grey, use_global=use_global )

    # Model
    Model_ij = np.dot(Kernel_il, np.dot(X_area_lk, X_albd_kj))

    # Chi-squared statistic
    Diff_ij = ( Obs_ij - Model_ij ) / Obsnoise_ij
    Chi2_i  = np.diag(np.dot( Diff_ij, Diff_ij.T ))
    chi2    = np.sum(Chi2_i)

    # Flat prior for albedo
    Y_albd_kj = Y_array[0:n_type*n_band].reshape([n_type, n_band])
    ln_prior_albd = prior.get_ln_prior_albd( Y_albd_kj )

    # Extract Y_area_lk from Y_array
    if use_grey and use_global:
        Y_area_lk = Y_array[(n_type-1)*n_band:(n_type-1)*n_band+n_slice*(n_type-2)].reshape([n_slice, n_type-2])
    elif use_grey and not use_global:
        Y_area_lk = Y_array[(n_type-1)*n_band:(n_type-1)*n_band+n_slice*(n_type-1)].reshape([n_slice, n_type-1])
    elif use_global and not use_grey:
        Y_area_lk = Y_array[n_type*n_band:n_type*n_band+n_slice*(n_type-2)].reshape([n_slice, n_type-2])
    else:
        Y_area_lk = Y_array[n_type*n_band:(n_type)*n_band+n_slice*(n_type-1)].reshape([n_slice, n_type-1])

    # flat prior for area fraction
    ln_prior_area = prior.get_ln_prior_area_new( Y_area_lk, X_area_lk[:,:-1] )

    # flat ordering prior for labeling degeneracy
    ln_prior_order = prior.get_ln_prior_ordering(X_albd_kj, X_area_lk)

    # atmosphere prior
    #ln_prior_atmosphere = prior.get_ln_prior_atmosphere(X_albd_kj, X_area_lk, use_grey, use_global, max_dev = max_dev)

    # regularization
    if regularization is not None:
        if ( regularization == 'Tikhonov' ):
            # ---Tikhonov Regularization
            regparam = Y_array[-1*n_regparam]
            regterm_area = prior.regularize_area_tikhonov( X_area_lk, regparam )
        elif ( regularization == 'GP' ):
            # ---Gaussian Process
            regparam = ( Y_array[-1*n_regparam], Y_array[-1*n_regparam+1], Y_array[-1*n_regparam+2] )
            regterm_area = prior.regularize_area_GP( X_area_lk, regparam )
        elif ( regularization == 'GP2' ):
            # ---Gaussian Process without constraint
            regparam = ( Y_array[-1*n_regparam], Y_array[-1*n_regparam+1] )
            regterm_area = prior.regularize_area_GP2( X_area_lk, regparam )
    # ---Others
    else :
        regterm_area = 0.

    # verbose
    if verbose :
        print 'chi2', chi2 - ln_prior_albd - ln_prior_area, chi2, ln_prior_albd, ln_prior_area
        print 'chi2/d.o.f.', chi2 / (len(Y_array)*1.-1.), len(Y_array)

    answer = - chi2 \
             + ln_prior_albd \
             + ln_prior_area \
             + ln_prior_order \
             + regterm_area

    # Check for nans
    if np.isnan(answer):
        answer = -np.inf

    if flip :
        return -1. * answer
    else :
         return answer, Model_ij

################################################################################

def lnlike(Y_array, *args):
    """
    Log-likelihood function for mapping

    Parameters
    ----------
    Y_array : numpy.ndarray

    Returns
    -------
    """

    # Unpack args
    Obs_ij, Obsnoise_ij, Kernel_il, regularization, n_regparam, flip, verbose, n_type, n_slice  = args
    n_band = len(Obs_ij[0])

    # Parameter conversion
    if (n_regparam > 0):
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X(Y_array[:-1*n_regparam], n_type, n_band, n_slice )
    else:
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X(Y_array, n_type, n_band, n_slice )

    # Model
    Model_ij = np.dot(Kernel_il, np.dot(X_area_lk, X_albd_kj))

    # Chi-squared statistic
    Diff_ij = ( Obs_ij - Model_ij ) / Obsnoise_ij
    Chi2_i  = np.diag(np.dot( Diff_ij, Diff_ij.T ))
    chi2    = np.sum(Chi2_i)

    # verbose
    if verbose :
        print 'chi2', chi2
        print 'chi2/d.o.f.', chi2 / (len(Y_array)*1.-1.), len(Y_array)

    answer = - chi2

    # Check for nans
    if np.isnan(answer):
        answer = -np.inf

    if flip :
        return -1. * answer
    else :
         return answer

def lnprior(Y_array, *args):
    """
    Log-Prior function for mapping

    Parameters
    ----------

    Returns
    -------
    """

    # Unpack args
    Obs_ij, Obsnoise_ij, Kernel_il, regularization, n_regparam, flip, verbose, n_type, n_slice  = args
    n_band = len(Obs_ij[0])

    # Parameter conversion
    if (n_regparam > 0):
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X(Y_array[:-1*n_regparam], n_type, n_band, n_slice )
    else:
        X_albd_kj, X_area_lk = reparameterize.transform_Y2X(Y_array, n_type, n_band, n_slice )

    # Flat prior for albedo
    Y_albd_kj = Y_array[0:n_type*n_band].reshape([n_type, n_band])
    ln_prior_albd = prior.get_ln_prior_albd( Y_albd_kj )

    # flat prior for area fraction
    Y_area_lk = Y_array[n_type*n_band:n_type*n_band+n_slice*(n_type-1)].reshape([n_slice, n_type-1])
    ln_prior_area = prior.get_ln_prior_area_new( Y_area_lk, X_area_lk[:,:-1] )

    # flat ordering prior for labeling degeneracy
    ln_prior_order = prior.get_ln_prior_ordering(X_albd_kj, X_area_lk)

    # regularization
    if regularization is not None:
        if ( regularization == 'Tikhonov' ):
            # ---Tikhonov Regularization
            regparam = Y_array[-1*n_regparam]
            regterm_area = prior.regularize_area_tikhonov( X_area_lk, regparam )
        elif ( regularization == 'GP' ):
            # ---Gaussian Process
            regparam = ( Y_array[-1*n_regparam], Y_array[-1*n_regparam+1], Y_array[-1*n_regparam+2] )
            regterm_area = prior.regularize_area_GP( X_area_lk, regparam )
        elif ( regularization == 'GP2' ):
            # ---Gaussian Process without constraint
            regparam = ( Y_array[-1*n_regparam], Y_array[-1*n_regparam+1] )
            regterm_area = prior.regularize_area_GP2( X_area_lk, regparam )
    # ---Others
    else :
        regterm_area = 0.

    answer = ln_prior_albd + ln_prior_area + ln_prior_order + regterm_area

    # Check for nans
    if np.isnan(answer):
        answer = -np.inf

    if flip :
        return -1. * answer
    else :
         return answer
