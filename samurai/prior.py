import numpy as np
import scipy as sp

__all__ = [
    "get_ln_prior_atmosphere",
    "get_ln_prior_ordering",
    "get_ln_prior_albd",
    "get_ln_prior_area_new",
    "get_ln_prior_area",
    "get_cov",
    "regularize_area_GP2",
    "get_cov2",
    "regularize_area_tikhonov"
]

def get_ln_prior_atmosphere(x_albd_kj, x_area_lk, use_grey, use_global, max_dev = 0.05):
    """
    This probably won't work becuase it will penalize any iterative attempt to
    move the baseline...
    """
    ln_prior = 0.0

    if use_grey and (not use_global):
        #
        m = np.mean(x_albd_kj[-1,:])
        dev = np.fabs(x_albd_kj[-1,:] - m)
        #diff = x_albd_kj[-1,:] - m
        #probs = sp.stats.norm(np.mean(x_albd_kj[-1,:]), max_dev).pdf(x_albd_kj[-1,:])
        #
        if np.any(dev > max_dev):
            ln_prior = np.inf
        else:
            ln_prior = 0.0
    elif use_global and (not use_grey):
        #
        dev = np.fabs(x_area_lk[:,-1] - np.mean(x_area_lk[:,-1]))
        #
        if np.any(dev > max_dev):
            ln_prior = np.inf
        else:
            ln_prior = 0.0
    elif use_grey and use_global:
        #
        dev1 = np.fabs(x_albd_kj[-2,:] - np.mean(x_albd_kj[-2,:]))
        #
        dev2 = np.fabs(x_area_lk[:,-1] - np.mean(x_area_lk[:,-1]))
        #
        if np.any(dev1 > max_dev) or np.any(dev2 > max_dev):
            ln_prior = np.inf
        else:
            ln_prior = 0.0
    else:
        ln_prior = 0.0

    return ln_prior


def get_ln_prior_ordering(x_albd_kj, x_area_lk):
    # Calculate "detectability" metric
    dm = np.mean(x_albd_kj, axis=1) * np.mean(x_area_lk, axis=0)
    # Sort by detectability
    dms = np.sort(dm)
    # Check sorted against original
    if False in dm == dms:
        # Had to resort, reject sample
        ln_prior = np.inf
    else:
        # Stays in order, accept sample
        ln_prior = 0.0
    return ln_prior

#---------------------------------------------------
def get_ln_prior_albd( y_albd_kj ):

    prior_kj = np.exp( y_albd_kj ) / ( 1 + np.exp( y_albd_kj ) )**2
    ln_prior = np.log( np.prod( prior_kj ) )
#    ln_prior = np.sum( np.log( prior_kj ) )
    return ln_prior


#---------------------------------------------------
def get_ln_prior_area_new( y_area_lk, x_area_lk ):

#    x_area_lk is a dummy
    prior_lk = np.exp( y_area_lk ) / ( 1 + np.exp( y_area_lk ) )**2
    ln_prior = np.log( np.prod( prior_lk ) )
    return ln_prior


#---------------------------------------------------
def get_ln_prior_area( y_area_lk, x_area_lk ):

    l_dim = len( y_area_lk )
    k_dim = len( y_area_lk.T )
    kk_dim = len( y_area_lk.T )

    expY_area_lk = np.exp( y_area_lk )
    expYY_area_lk = 1./( 1 + expY_area_lk )
    cumprodY_area_lk = np.cumprod( expYY_area_lk, axis=1 )

    # when kk < k
    dFdg = np.zeros( [ l_dim, k_dim, kk_dim  ] )
    l_indx, k_indx, kk_indx = np.meshgrid( np.arange( l_dim ), np.arange( k_dim ), np.arange( kk_dim ), indexing='ij' )
#    print ''
#    print l_indx, k_indx, kk_indx
#    print ''
    dFdg[ l_indx, k_indx, kk_indx ] = -1. * x_area_lk[ l_indx, kk_indx ] * expY_area_lk[ l_indx, k_indx ] * cumprodY_area_lk[ l_indx, k_indx ]

    # when kk > k
    k_tmp, kk_tmp   = np.triu_indices(k_dim)
    l_indx, k_indx  = np.meshgrid( np.arange( l_dim ), k_tmp,  indexing='ij' )
    l_indx, kk_indx = np.meshgrid( np.arange( l_dim ), kk_tmp, indexing='ij' )
    dFdg[ l_indx, k_indx, kk_indx ] = 0.

    # when kk = k
    k_tmp, kk_tmp = np.diag_indices(k_dim)
    l_indx, k_indx  = np.meshgrid( np.arange( l_dim ), k_tmp,  indexing='ij' )
    l_indx, kk_indx = np.meshgrid( np.arange( l_dim ), kk_tmp, indexing='ij' )
    dFdg[ l_indx, k_indx, kk_indx ] = x_area_lk[ l_indx, k_indx ] * cumprodY_area_lk[ l_indx, kk_indx ]

    ln_prior = np.sum( np.log( np.linalg.det( dFdg ) ) )

    return ln_prior



##---------------------------------------------------
#def get_ln_prior_albd_new( y_albd_kj ):
#
#    dAdb = 1. / ( 1 + np.exp( y_albd_kj ) )**2
#    ln_prior = np.log( np.prod( dAdb ) )
#    return ln_prior
#

#---------------------------------------------------
def get_ln_prior_area_old( y_area_lk, x_area_lk ):

    l_dim = len( y_area_lk )
    k_dim = len( y_area_lk.T )
    kk_dim = len( y_area_lk.T )

    sumF = np.cumsum( x_area_lk, axis=1 )

    # when kk < k
    l_indx, k_indx, kk_indx = np.meshgrid( np.arange( l_dim ), np.arange( k_dim ), np.arange( kk_dim ), indexing='ij' )
    dgdF = np.zeros( [ l_dim, k_dim, kk_dim  ] )
    dgdF[ l_indx, k_indx, kk_indx ] = x_area_lk[ l_indx, kk_indx ] / x_area_lk[ l_indx, k_indx ] / ( 1 - sumF[ l_indx, k_indx ] )

    # when kk > k
    k_tmp, kk_tmp   = np.triu_indices(k_dim)
    l_indx, k_indx  = np.meshgrid( np.arange( l_dim ), k_tmp,  indexing='ij' )
    l_indx, kk_indx = np.meshgrid( np.arange( l_dim ), kk_tmp, indexing='ij' )
    dgdF[ l_indx, k_indx, kk_indx ] = 0.

    # when kk = k
    k_tmp, kk_tmp = np.diag_indices(k_dim)
    l_indx, k_indx  = np.meshgrid( np.arange( l_dim ), k_tmp,  indexing='ij' )
    l_indx, kk_indx = np.meshgrid( np.arange( l_dim ), kk_tmp, indexing='ij' )
    dgdF[ l_indx, k_indx, kk_indx ] = 1./x_area_lk[l_indx,k_indx]*(1. - sumF[l_indx, k_indx-1]) / ( 1 - sumF[ l_indx, k_indx ] )

    det_dgdF = np.linalg.det( dgdF )
    ln_prior = np.log( np.prod( 1./det_dgdF ) )

    return ln_prior



#---------------------------------------------------
def regularize_area_GP( x_area_lk, regparam ):

    sigma, wn_rel_amp_seed, lambda_angular = regparam

    wn_rel_amp = np.exp( wn_rel_amp_seed ) / ( 1. + np.exp( wn_rel_amp_seed ) )
#    print 'wn_rel_amp', wn_rel_amp
#    print 'lambda_angular', lambda_angular
    l_dim = len( x_area_lk )
    cov = get_cov( sigma, wn_rel_amp, lambda_angular, l_dim )

#    print 'cov', cov
    #inv_cov = np.linalg.inv( cov )
    det_cov = np.linalg.det( cov )
    #if ( det_cov == 0. ):
    #    print 'det_cov', det_cov
    #    print 'cov', cov

    x_area_ave = 1./len(x_area_lk.T)
    dx_area_lk = x_area_lk[:,:-1] - x_area_ave
    try:
        term1_all = np.dot( dx_area_lk.T, np.linalg.solve( cov, dx_area_lk ) ) #np.dot( dx_area_lk.T, np.dot( inv_cov, dx_area_lk ) )
        term1 = -0.5 * np.sum( term1_all.diagonal() )
    except np.linalg.linalg.LinAlgError:
        term1 = -np.inf
    term2 = -0.5 * np.log( det_cov )

    prior_wn_rel_amp = np.log( wn_rel_amp / ( 1. + np.exp( wn_rel_amp_seed ) )**2 )

    return term1 + term2 + prior_wn_rel_amp


#---------------------------------------------------
def get_cov( sigma, wn_rel_amp, lambda_angular, l_dim, periodic=True):

#    kappa0 = np.log(output["x"][-1]) - np.log(360.0 - output["x"][-1])
    Sigma_ll = np.zeros([l_dim, l_dim])
    lon_l = 2.0 * np.pi * np.arange( l_dim ) / ( l_dim * 1. )
    dif_lon_ll = lon_l[:,np.newaxis] - lon_l[np.newaxis,:]
    if periodic :
        dif_lon_ll = np.minimum( abs( dif_lon_ll ), abs( 2. * np.pi - dif_lon_ll ) )
    else :
        dif_lon_ll = abs( dif_lon_ll )

#    Sigma_ll = np.exp( - 0.5 * dif_lon_ll**2 / ( lambda_angular**2 ) )
    Sigma_ll = np.exp( - dif_lon_ll / ( lambda_angular**2 ) )

    cov = Sigma_ll * ( 1 - wn_rel_amp )
    cov[np.diag_indices(l_dim)] += wn_rel_amp
    cov /= (1.0 +  wn_rel_amp)
    cov = cov * sigma

    return cov


#---------------------------------------------------
def regularize_area_GP2( x_area_lk, regparam ):

    sigma, lambda_angular = regparam

#    print 'wn_rel_amp', wn_rel_amp
#    print 'lambda_angular', lambda_angular
    l_dim = len( x_area_lk )
    cov = get_cov2( sigma, lambda_angular, l_dim )

#    print 'cov', cov
    inv_cov = np.linalg.inv( cov )
    det_cov = np.linalg.det( cov )

    print 'det_cov', det_cov
    print 'cov', cov
    if ( det_cov == 0. ):
        print 'det_cov', det_cov
        print 'cov', cov


#    print 'inv_cov', inv_cov
    x_area_ave = 1./len(x_area_lk.T)
    dx_area_lk = x_area_lk[:,:-1] - x_area_ave
    term1_all = np.dot( dx_area_lk.T, np.dot( inv_cov, dx_area_lk ) )
    term1 = -0.5 * np.sum( term1_all.diagonal() )
    term2 = -0.5 * np.log( det_cov )

#    prior_wn_rel_amp = np.log( wn_rel_amp / ( 1. + np.exp( wn_rel_amp_seed ) ) )

    return term1 + term2


#---------------------------------------------------
def get_cov2( sigma, lambda_angular, l_dim, periodic=True):

#    kappa0 = np.log(output["x"][-1]) - np.log(360.0 - output["x"][-1])
    Sigma_ll = np.zeros([l_dim, l_dim])
    lon_l = 2.0 * np.pi * np.arange( l_dim ) / ( l_dim * 1. )
    dif_lon_ll = lon_l[:,np.newaxis] - lon_l[np.newaxis,:]
    if periodic :
        dif_lon_ll = np.minimum( abs( dif_lon_ll ), abs( 2. * np.pi - dif_lon_ll ) )
    else :
        dif_lon_ll = abs( dif_lon_ll )

    Sigma_ll = np.exp( - 0.5 * dif_lon_ll**2 / ( lambda_angular**2 ) )
#    Sigma_ll = np.exp( - dif_lon_ll / ( lambda_angular**2 ) )

    cov = Sigma_ll * sigma
#    cov[np.diag_indices(l_dim)] = 0.

    return cov



#---------------------------------------------------
def regularize_area_tikhonov( x_area_lk, regparam ):

    lmd = regparam
    x_area_ave = 1./(len( x_area_lk.T ))
    dx_area_lk = x_area_lk[:,:-1] - x_area_ave
    term = -1.*np.sum( dx_area_lk**2. )/lmd**2
    prior_lmd = -0.5 * np.log( abs(lmd) )
    return term + prior_lmd


##---------------------------------------------------
#def get_ln_prior_albd_old( y_albd_kj ):
#
#    ln_prior = 0.
#    for k in xrange( len(y_albd_kj) ):
#        for j in xrange( len(y_albd_kj.T) ):
#            yy = y_albd_kj[k,j]
#             prior = np.exp( yy ) / ( 1 + np.exp( yy ) )**2
#            if ( prior > 0. ):
#                ln_prior = ln_prior + np.log( prior )
#            else:
 #                print "ERROR! ln_prior_albd is NaN"
#                print "  y, prior   ", yy, prior
#                ln_prior = ln_prior + 0.0
#
#    return ln_prior



#---------------------------------------------------
#def get_ln_prior_area_old( y_area_lj, x_area_lj ):
#
#    dydx_det = 1.
#    for ll in xrange( len( y_area_lj ) ):
#        dydx = np.zeros( [ len( y_area_lj.T ), len( y_area_lj.T ) ] )
#        for ii in xrange( len( dydx ) ):
#            jj = 0
#            # jj < ii
#            while ( jj < ii ):
#                g_i = y_area_lj[ll,ii]
#                f_i = x_area_lj[ll,ii]
#                f_j = x_area_lj[ll,jj]
#                sum_fi = np.sum( x_area_lj[ll,:ii+1] )
#                dydx[ii][jj] = 1. / ( 1. - sum_fi )
#                jj = jj + 1
#            # jj == ii
#            g_i = y_area_lj[ll,ii]
#            f_i = x_area_lj[ll,ii]
#            f_j = x_area_lj[ll,jj]
#            sum_fi = np.sum( x_area_lj[ll,:ii+1] )
#            dydx[ii][jj] = 1. / f_i * ( 1. - sum_fi + f_i ) / ( 1 - sum_fi )
#
##        print "dydx", dydx
##        print "det", np.linalg.det( dydx )
#        dydx_det = dydx_det * np.linalg.det( dydx )
#    dxdy_det = 1. / dydx_det
#
#    if ( dxdy_det <= 0. ):
#        print "ERROR! ln_prior_area is NaN"
#        print "     ", dxdy_det
#        sys.exit()
#
#    ln_prior = np.log( dxdy_det )
#    return ln_prior
