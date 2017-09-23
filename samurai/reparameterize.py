import numpy as np

#---------------------------------------------------
def transform_Y2X(Y_array, n_type, n_band, n_slice, flatten=False):

    Y_array = np.maximum(Y_array, -10)
    Y_array = np.minimum(Y_array, 10)

    # 'albedo' part
    Y_albd_kj_flat = Y_array[0:n_type*n_band]
    X_albd_kj_flat = np.exp( Y_albd_kj_flat )/( 1 + np.exp( Y_albd_kj_flat ) )

    # 'area fraction' part
    Y_area_lk = Y_array[n_type*n_band:n_type*n_band+n_slice*(n_type-1)].reshape([n_slice, n_type-1])
    expY_area_lk = np.exp( Y_area_lk )
    expYY_area_lk = 1./( 1 + expY_area_lk )
    cumprodY_area_lk = np.cumprod( expYY_area_lk, axis=1 )
    X_area_lk = expY_area_lk * cumprodY_area_lk

#    print '1. X_area_lk', X_area_lk

    X_area_lk = np.c_[X_area_lk, 1. - np.sum( X_area_lk, axis=1 )]
    X_area_lk_flat = X_area_lk.flatten()

#    print '2. X_area_lk', X_area_lk

    if flatten :
        X_array = np.r_[ X_albd_kj_flat, X_area_lk_flat ]
        return X_array
    else :
        X_albd_kj = X_albd_kj_flat.reshape([ n_type, n_band ])
        return X_albd_kj, X_area_lk


#---------------------------------------------------
def transform_X2Y(X_albd_kj, X_area_lk):

    n_slice = len( X_area_lk )
    n_type  = len( X_albd_kj )

    # 'albedo' part
    Y_albd_kj = np.log(X_albd_kj) - np.log(1.-X_albd_kj)

    # 'area' part
    sumX_area_lk = np.cumsum( X_area_lk, axis=1 )
    Y_area_lk = np.log( X_area_lk[:,:-1] / ( 1. - sumX_area_lk[:,:-1] ) )

    return np.concatenate([Y_albd_kj.flatten(), Y_area_lk.flatten()])

#---------------------------------------------------
def transform_Y2X_atmosphere(Y_array, n_type, n_band, n_slice, use_grey = False, use_global = False, flatten=False):
    """
    Transform state vector back into physical parameters
    """

    if (not use_grey) and (not use_global):
        # original model

        # 'albedo' part
        Y_albd_kj_flat = Y_array[0:n_type*n_band]
        X_albd_kj_flat = np.exp( Y_albd_kj_flat )/( 1 + np.exp( Y_albd_kj_flat ) )

        # 'area fraction' part
        Y_area_lk = Y_array[n_type*n_band:n_type*n_band+n_slice*(n_type-1)].reshape([n_slice, n_type-1])
        expY_area_lk = np.exp( Y_area_lk )
        expYY_area_lk = 1./( 1 + expY_area_lk )
        cumprodY_area_lk = np.cumprod( expYY_area_lk, axis=1 )
        X_area_lk = expY_area_lk * cumprodY_area_lk

        X_area_lk = np.c_[X_area_lk, 1. - np.sum( X_area_lk, axis=1 )]
        X_area_lk_flat = X_area_lk.flatten()

    elif use_grey and (not use_global):

        # 'albedo' part
        Y_albd_kj_flat = Y_array[0:(n_type-1)*n_band]
        X_albd_kj_flat = np.exp( Y_albd_kj_flat )/( 1 + np.exp( Y_albd_kj_flat ) )

        # Treat grey albedo
        Y_albd_grey = Y_array[-1]
        X_albd_grey = np.exp( Y_albd_grey )/( 1 + np.exp( Y_albd_grey ) )
        X_albd_kj_flat = np.hstack([X_albd_kj_flat, X_albd_grey * np.ones(n_band)])

        # 'area fraction' part
        Y_area_lk = Y_array[(n_type-1)*n_band:(n_type-1)*n_band+n_slice*(n_type-1)].reshape([n_slice, n_type-1])
        expY_area_lk = np.exp( Y_area_lk )
        expYY_area_lk = 1./( 1 + expY_area_lk )
        cumprodY_area_lk = np.cumprod( expYY_area_lk, axis=1 )
        X_area_lk = expY_area_lk * cumprodY_area_lk

        X_area_lk = np.c_[X_area_lk, 1. - np.sum( X_area_lk, axis=1 )]
        X_area_lk_flat = X_area_lk.flatten()

    elif use_global and (not use_grey):

        # 'albedo' part
        Y_albd_kj_flat = Y_array[0:n_type*n_band]
        X_albd_kj_flat = np.exp( Y_albd_kj_flat )/( 1 + np.exp( Y_albd_kj_flat ) )

        # Treat global area fraction
        Y_area_global = Y_array[-1]
        X_area_global = np.exp( Y_area_global )/( 1 + np.exp( Y_area_global ) )

        # 'area fraction' part
        Y_area_lk = Y_array[n_type*n_band:n_type*n_band+n_slice*(n_type-2)].reshape([n_slice, n_type-2])
        expY_area_lk = np.exp( Y_area_lk )
        expYY_area_lk = (1.0 - X_area_global)/( 1.0 + expY_area_lk )
        cumprodY_area_lk = np.cumprod( expYY_area_lk, axis=1 )
        X_area_lk = expY_area_lk * cumprodY_area_lk

        X_area_lk = np.c_[X_area_lk, (1.0 - X_area_global) - np.sum( X_area_lk, axis=1 ), X_area_global*np.ones(n_slice)]
        X_area_lk_flat = X_area_lk.flatten()

    elif use_grey and use_global:

        # 'albedo' part
        Y_albd_kj_flat = Y_array[0:(n_type-1)*n_band]
        X_albd_kj_flat = np.exp( Y_albd_kj_flat )/( 1 + np.exp( Y_albd_kj_flat ) )

        # Treat grey albedo (second to last item)
        Y_albd_grey = Y_array[-2]
        X_albd_grey = np.exp( Y_albd_grey )/( 1 + np.exp( Y_albd_grey ) )

        # Combine albedos and Swap last two columns in albedo grid because grey
        # alb is second to last when global model is also used
        X_albd_kj_flat_new = np.zeros(n_band*n_type)
        X_albd_kj_flat_new[0:n_band*(n_type-2)] = X_albd_kj_flat[0:n_band*(n_type-2)]
        X_albd_kj_flat_new[n_band*(n_type-2):n_band*(n_type-1)] = X_albd_grey * np.ones(n_band)
        X_albd_kj_flat_new[n_band*(n_type-1):] = X_albd_kj_flat[n_band*(n_type-2):]

        #X_albd_kj_flat = np.hstack([X_albd_kj_flat, X_albd_grey * np.ones(n_band)])
        X_albd_kj_flat = X_albd_kj_flat_new

        # Treat global area fraction (last item)
        Y_area_global = Y_array[-1]
        X_area_global = np.exp( Y_area_global )/( 1 + np.exp( Y_area_global ) )

        # 'area fraction' part
        Y_area_lk = Y_array[(n_type-1)*n_band:(n_type-1)*n_band+n_slice*(n_type-2)].reshape([n_slice, n_type-2])
        expY_area_lk = np.exp( Y_area_lk )
        expYY_area_lk = (1.0 - X_area_global)/( 1.0 + expY_area_lk )
        cumprodY_area_lk = np.cumprod( expYY_area_lk, axis=1 )
        X_area_lk = expY_area_lk * cumprodY_area_lk

        X_area_lk = np.c_[X_area_lk, (1.0 - X_area_global) - np.sum( X_area_lk, axis=1 ), X_area_global*np.ones(n_slice)]
        X_area_lk_flat = X_area_lk.flatten()

    if flatten :
        X_array = np.r_[ X_albd_kj_flat, X_area_lk_flat ]
        return X_array
    else :
        X_albd_kj = X_albd_kj_flat.reshape([ n_type, n_band ])
        return X_albd_kj, X_area_lk


#---------------------------------------------------
def transform_X2Y_atmosphere(X_albd_kj, X_area_lk, use_grey = False, use_global = False):
    """
    Transforms physical X array to parameterized Y array for fitting (state vector)
    """

    n_slice = len( X_area_lk )
    n_type  = len( X_albd_kj )

    if (not use_grey) and (not use_global):

        # 'albedo' part
        Y_albd_kj = np.log(X_albd_kj) - np.log(1.-X_albd_kj)

        # 'area' part
        sumX_area_lk = np.cumsum( X_area_lk, axis=1 )
        Y_area_lk = np.log( X_area_lk[:,:-1] / ( 1. - sumX_area_lk[:,:-1] ) )

        # Compose Y_array
        Y_array = np.concatenate([Y_albd_kj.flatten(), Y_area_lk.flatten()])

    elif use_grey and (not use_global):

        # Get grey albedo
        X_albd_grey = X_albd_kj[-1,:][0]

        # Delete grey albedo row
        X_albd_new = np.delete(X_albd_kj, -1, 0)

        # Reparameterize albedo
        Y_albd_kj = np.log(X_albd_new) - np.log(1.-X_albd_new)
        Y_albd_grey = np.log(X_albd_grey) - np.log(1.-X_albd_grey)

        # Reparameterize area
        sumX_area_lk = np.cumsum( X_area_lk, axis=1 )
        Y_area_lk = np.log( X_area_lk[:,:-1] / ( 1. - sumX_area_lk[:,:-1] ) )

        # Compose Y_array
        Y_array = np.hstack([Y_albd_kj.flatten(), Y_area_lk.flatten(), Y_albd_grey])

    elif use_global and (not use_grey):

        # Get global area
        X_area_global = X_area_lk[:,-1][0]

        # cumulative sum except last column
        sumX_area_new = np.cumsum( X_area_lk[:,:-1], axis=1 )
        # reparameterize areas except last two columns
        Y_area_lk = np.log( X_area_lk[:,:-2] / ( (1.0 - X_area_global) - sumX_area_new[:,:-1] ) )
        # reparameterize lone area
        Y_area_global = np.log(X_area_global) - np.log(1.-X_area_global)

        # Reparameterize albedo
        Y_albd_kj = np.log(X_albd_kj) - np.log(1.-X_albd_kj)

        # Compose Y_array
        Y_array = np.hstack([Y_albd_kj.flatten(), Y_area_lk.flatten(), Y_area_global])

    elif use_grey and use_global:

        # Get grey albedo
        X_albd_grey = X_albd_kj[-2,:][0]

        # Delete grey albedo row
        X_albd_new = np.delete(X_albd_kj, -2, 0)

        # Reparameterize albedo
        Y_albd_kj = np.log(X_albd_new) - np.log(1.-X_albd_new)
        Y_albd_grey = np.log(X_albd_grey) - np.log(1.-X_albd_grey)

        # Get global area
        X_area_global = X_area_lk[:,-1][0]

        # cumulative sum except last column
        sumX_area_new = np.cumsum( X_area_lk[:,:-1], axis=1 )
        # reparameterize areas except last two columns
        Y_area_lk = np.log( X_area_lk[:,:-2] / ( 1.0 - X_area_global - sumX_area_new[:,:-1] ) )
        # reparameterize lone area
        Y_area_global = np.log(X_area_global) - np.log(1.-X_area_global)

        # Compose Y_array
        Y_array = np.hstack([Y_albd_kj.flatten(), Y_area_lk.flatten(), Y_albd_grey, Y_area_global])

    return Y_array
