import numpy as np
import h5py

#---------------------------------------------------

def set_nreg(reg):
    """
    Set number of extra parameters due to regularization
    """
    if reg is not None:
        if reg == 'Tikhonov':
            N = 1
        elif reg == 'GP':
            N = 3
        elif reg == 'GP2':
            N = 2
        else:
            print("%s is not a valid regularization method. Using no regularization." %reg)
            N = 0
    else:
        N = 0
    return N

#---------------------------------------------------

def save2hdf5(f, dataset, name="dataset", dictionary=None, compression='lzf', close=True):
    """
    Saves a dataset (e.g. numpy array) to a new or existing hdf5 file with ability to save a
    dictionary as attributes to that dataset.

    Parameters
    ----------
    f : str OR h5py._hl.files.File
        Name of new hdf5 file OR existing h5py file object
    dataset : numpy array
        Data to be saved
    name : str
        Name of new dataset to be saved
    dictionary : dict
        Dictionary of attributes for dataset
    compression : str
        h5py compression algorithm
    close : bool
        If False returns new h5py._hl.files.File
    """

    # Hack for interactive use with iPython notebook:
    try:
        f.close()
    except (NameError, ValueError, AttributeError):
        pass

    # Create new hdf5 file
    if type(f) is str:
        filename = f
        f = h5py.File(filename, 'w')
    elif type(f) is h5py._hl.files.File:
        pass
    else:
        print "Invalid type(f)"
        return None

    # Create dataset for mcmc chain samples
    s = f.create_dataset(name, data=dataset, compression=compression)

    # Save dictionary to attributes
    if dictionary is not None:
        # Loop through python dictionary, saving each item as an hdf5 attribute
        # under the same name
        for key, value in dictionary.iteritems(): s.attrs[key] = value

    if close:
        # Close the file stream
        f.close()
        return None
    else:
        # Return the file stream
        return f
#---------------------------------------------------

#---------------------------------------------------
def generate_tex_names(n_type, n_band, n_slice):
    """
    Generate an array of Latex strings for each parameter in the
    X and Y vectors.

    Returns
    -------
    Y_names : array
        Non-physical fitting parameters
    X_names : array
        Physical parameters for Albedo and Surface Area Fractions
    """
    # Create list of strings for Y parameter names
    btmp = []
    gtmp = []
    for i in range(n_type):
        for j in range(n_band):
            btmp.append(r"b$_{"+str(i+1)+","+str(j+1)+"}$")
    for j in range(n_type - 1):
        for i in range(n_slice):
            gtmp.append(r"g$_{"+str(i+1)+","+str(j+1)+"}$")
    Y_names = np.concatenate([np.array(btmp), np.array(gtmp)])

    # Create list of strings for X parameter names
    Atmp = []
    Ftmp = []
    for i in range(n_type):
        for j in range(n_band):
            Atmp.append(r"A$_{"+str(i+1)+","+str(j+1)+"}$")
    for j in range(n_type):
        for i in range(n_slice):
            Ftmp.append(r"F$_{"+str(i+1)+","+str(j+1)+"}$")
    X_names = np.concatenate([np.array(Atmp), np.array(Ftmp)])

    return Y_names, X_names

#---------------------------------------------------

def calculate_walkers(n_dim):
    return 10*n_dim

#---------------------------------------------------
