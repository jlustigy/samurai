import numpy as np

#--------------------------------------------------------------------
# Parameters
#--------------------------------------------------------------------

NUM_MCMC = 10000
NUM_MCMC_BURNIN = 0
SEED_AMP = 0.5

#REGULARIZATION = None
REGULARIZATION = 'GP'
#REGULARIZATION = 'GP2'
#REGULARIZATION = 'Tikhonov'

SIGMA_Y  = 3.0
NOISELEVEL = 0.01

FLAG_REG_AREA = False
FLAG_REG_ALBD = False

#n_slice = 4
N_TYPE  = 2

deg2rad = np.pi/180.

N_SIDE   = 32

#INFILE = "data/raddata_12_norm"
##INFILE = "data/raddata_2_norm"
#INFILE = "mockdata/mock_simple_1_data"
#INFILE = "mockdata/mock_simple_3types_1_data"
# INFILE = 'mockdata/mock_simple_1_scattered0.01_data_with_noise'
INFILE = 'mockdata/simpleIGBP_quadrature_lc'
WAVEBAND_CENTERS = np.array([550., 650., 750., 850.])
WAVEBAND_WIDTHS = np.array([100., 100., 100., 100.])

HDF5_COMPRESSION = 'lzf'

def calculate_walkers(n_dim):
    return 10*n_dim
