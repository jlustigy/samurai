import numpy as np
import healpy as hp
import pdb

__all__ = ["latlon2cart", "weight", "kernel"]

deg2rad = (np.pi/180.)

#---------------------------------------------------
def latlon2cart(lat, lon):
    x = np.sin((90.-lat)*deg2rad)*np.cos(lon*deg2rad)
    y = np.sin((90.-lat)*deg2rad)*np.sin(lon*deg2rad)
    z = np.cos((90.-lat)*deg2rad)*(0.*lon) # (0.*lon) is a dummy
    return np.dstack([x,y,z])[0]


#---------------------------------------------------
def weight(time, n_side, param_geometry):
    """
    Weight of pixel at (lat_r, lon_r) assuming Lambertian surface
    """
    lat_o, lon_o, lat_s, lon_s, omega = param_geometry
    EO_vec = latlon2cart(lat_o, lon_o-omega*time/deg2rad)
    ES_vec = latlon2cart(lat_s, lon_s-omega*time/deg2rad)
    ER_vec_array = np.array(hp.pix2vec(n_side, np.arange(hp.nside2npix(n_side))))
    cosTH0_array = np.dot(ES_vec, ER_vec_array)
    cosTH1_array = np.dot(EO_vec, ER_vec_array)
    return np.clip(cosTH0_array, 0., 1.)*np.clip(cosTH1_array, 0., 1.)


#---------------------------------------------------
#def weight(time, n_side, nn, param_geometry):
#    """
#    Weight of pixel at (lat_r, lon_r) assuming Lambertian surface
#    """
#    lat_o, lon_o, lat_s, lon_s, omega = param_geometry
#    EO_vec = latlon2cart(lat_o, lon_o-omega*time/deg2rad)
#    ES_vec = latlon2cart(lat_s, lon_s-omega*time/deg2rad)
#    ER_vec_array = np.array(hp.pix2vec(n_side, nn))
#    cosTH0_array = np.dot(ES_vec, ER_vec_array)
#    cosTH1_array = np.dot(EO_vec, ER_vec_array)
#    return np.clip(cosTH0_array, 0., 1.)*np.clip(cosTH1_array, 0., 1.)
#

##---------------------------------------------------
def kernel(Time_i, n_slice, n_side, param_geometry):
    """
    Kernel!
    """

    Weight_in = weight( Time_i, n_side, param_geometry )

    n_pix = hp.nside2npix( n_side )
    theta_n, phi_n   = hp.pixelfunc.pix2ang( n_side, np.arange(n_pix) )
    phi_n[ phi_n < np.pi ]  = phi_n[ phi_n < np.pi ] + 2. * np.pi
    assignedL_float_n = np.trunc(( phi_n - np.pi)/(2.*np.pi/n_slice))
    assignedL_n = assignedL_float_n.astype(np.int64)
    LN_nl = np.zeros([n_pix, n_slice])
    LN_nl[ np.arange(n_pix), assignedL_n ] = 1
    Kernel_il = np.dot( Weight_in, LN_nl )
    Kernel_il = Kernel_il / np.tile( np.sum( Kernel_il, axis=1 ), [ n_slice, 1 ] ).T

    return Kernel_il


#---------------------------------------------------
def kernel_old(Time_i, n_slice, n_side, param_geometry):
    """
    Kernel!
    """
    Kernel_il = np.zeros([len(Time_i), n_slice])
    n_pix = hp.nside2npix( n_side )
    position_theta, position_phi   = hp.pixelfunc.pix2ang( n_side, np.arange(n_pix))
    position_phi[ position_phi < np.pi ]  = position_phi[ position_phi < np.pi ] + 2. * np.pi
    assignedL_float = np.trunc((position_phi - np.pi)/(2.*np.pi/n_slice))
    assignedL = assignedL_float.astype(np.int64)

    for ii in xrange(len(Time_i)):
        Weight_n = weight(Time_i[ii], n_side, param_geometry)
        Count_l = np.zeros(n_slice)
        for nn in xrange(n_pix):
            Kernel_il[ii][assignedL[nn]] += Weight_n[nn]
            Count_l[assignedL[nn]]       += 1
        Kernel_il[ii] = Kernel_il[ii]/Count_l
        Kernel_il[ii] = Kernel_il[ii]/np.sum(Kernel_il[ii])

    return Kernel_il
