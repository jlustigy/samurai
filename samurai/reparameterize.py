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

