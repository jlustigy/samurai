from __future__ import division, print_function, absolute_import, unicode_literals

# Import standard libraries
import numpy as np
import sys, imp, os
import datetime
import multiprocessing
from scipy.optimize import minimize
from types import ModuleType, FunctionType, StringType
from pdb import set_trace as stop

# Import dependent modules
import healpy as hp
import emcee
import emcee3
import h5py

# Import packages files
sys.path.insert(1, os.path.dirname(os.path.abspath(__file__)))
from . import geometry
from . import prior
from . import reparameterize
from .likelihood import lnprob, lnlike, lnprior, lnprob_atmosphere
from .map_utils import generate_tex_names, save2hdf5, calculate_walkers, set_nreg
from .mcmc_physical import plot_area_alb, plot_model_data, plot_posteriors
from .mcmc_analysis import plot_trace

__all__ = ["Mapper", "Data", "Output"]

# The location to *this* file
RELPATH = os.path.dirname(__file__)

################################################################################
# Output
################################################################################

class Output(object):
    """
    ``samurai`` simulation data are stored in this object which interfaces with the
    HDF5 file
    """
    def __init__(self, hpath=None):
        """
        Initialize a ``samurai`` Output object using the name of an HDF5 file

        Parameters
        ----------
        hpath : str
            Location of output HDF5 file
        """
        self.hpath=hpath
        self.open=self._open

    def _open(self, verbose=True):
        """Open HDF5 output file stream for file locatation stored in ``hpath``
        """
        # Try to open HDF5 file
        try:
            # Open the file stream
            f = h5py.File(self.hpath, 'r+')
            # Set n_slice based on forward model
            if f.attrs["fmodel"] == "map":
                nslice = f.attrs["nslice"]
            elif f.attrs["fmodel"] == "lightcurve":
                nslice = len(f["data/Obs_ij"])
            else:
                print("Error: %s is an unrecognized forward model" %f.attrs["fmodel"])
                return None
            # Put all the n's in a dictionary for easy access
            N = {
                "ntype" : f.attrs["ntype"],
                "nslice" : nslice,
                "nwalkers" : f["mcmc/samples"].shape[0],
                "nsteps" : f["mcmc/samples"].shape[1],
                "nparam" : f["mcmc/samples"].shape[2],
                "ntimes" : len(f["data/Obs_ij"]),
                "nband" : len(f["data/Obs_ij"][0]),
                "nregparam" : f.attrs["nregparam"]
            }
            self.N=N
            # Create new attribute for file stream
            self.hfile=f
            # Allow access to methods
            self.close=self._close
            self.plot_trace=self._plot_trace
            self.transform_samples=self._transform_samples
            self.plot_posteriors=self._plot_posteriors
            self.plot_area_alb=self._plot_area_alb
            self.plot_model_data=self._plot_model_data
            # Restrict access to open
            del self.open
            if verbose: print("HDF5 file opened")
        except IOError:
            print("Error: HDF5 file does not exist as %s" %self.hpath)

    def _close(self, verbose=True):
        """Close HDF5 output file stream for file locatation stored in ``hpath``
        """
        if hasattr(self, "hfile"):
            # Close HDF5 file stream
            self.hfile.close()
            # Delete file stream attribute
            del self.hfile
            # Delete close method attributes
            del self.close
            del self.plot_trace
            del self.transform_samples
            del self.plot_posteriors
            del self.plot_area_alb
            del self.plot_model_data
            # Allow access to open
            self.open=self._open
            if verbose: print("HDF5 file closed")

    def _plot_trace(self, verbose=True, which=None, newdir="trace_plots/"):
        """Plot MCMC trace plots
        """
        mdir = os.path.split(self.hpath)[0]
        # Create directory for trace plots
        trace_dir = os.path.join(mdir, newdir)
        try:
            os.mkdir(trace_dir)
            if verbose: print("Created directory:", trace_dir)
        except OSError:
            if verbose: print(trace_dir, "already exists.")
        # Make trace plots
        samples = self.hfile["mcmc/samples"]
        Y_names = self.hfile["mcmc"].attrs["Y_names"]
        plot_trace(samples, names=Y_names, directory=trace_dir, which=which)

    def _transform_samples(self, iburn, verbose=True, newname="physical_samples"):
        """Transform re-parameterized samples to physically meaningful units

        Parameters
        ----------
        iburn : int
            Burn-in index
        verbose : bool, optional
            Set to print status updates
        newname : str, optional
            Name of physical samples dataset in hdf5 file
        """

        f = self.hfile
        samples = self.hfile["mcmc/samples"]
        n_type = f.attrs["ntype"]
        nwalkers = samples.shape[0]
        nsteps = samples.shape[1]
        nparam = samples.shape[2]
        Obs_ij = f["data/Obs_ij"]
        n_times = len(Obs_ij)
        n_band = len(Obs_ij[0])
        n_regparam = f.attrs["nregparam"]

        # Set n_slice based on forward model
        if f.attrs["fmodel"] == "map":
            n_slice = f.attrs["nslice"]
        elif f.attrs["fmodel"] == "lightcurve":
            n_slice = len(Obs_ij)
        else:
            print("Error: %s is an unrecognized forward model" %f.attrs["fmodel"])
            return None

        # Throw assertion error if burn-in index exceeds number of steps
        assert iburn < samples.shape[1]

        # If the xsamples are already in the hdf5 file
        if newname in f["mcmc/"].keys():
            # load physical samples
            xs = f["mcmc/"+newname]
            if (xs.attrs["iburn"] == iburn) and (int(np.sum(xs[0,:])) != 0):
                # This is the exact same file or it has been loaded with 0's
                if verbose: print(newname + " loaded from file!")
                rerun = False
            else:
                # Must re-run xsamples with new burnin, overwrite
                if verbose: print("Different burn-in index here. Must reflatten and convert...")
                rerun = True

        # If the xsamples are not in the hdf5 file,
        # or if they need to be re-run
        if newname not in f["mcmc/"].keys() or rerun:

            # Determine shape of new dataset
            if (n_regparam > 0):
                nxparam = len(reparameterize.transform_Y2X(samples[0,0,:-1*n_regparam], n_type, n_band, n_slice, flatten=True))
            else:
                nxparam = len(reparameterize.transform_Y2X(samples[0,0,:], n_type, n_band, n_slice, flatten=True))
            new_shape = (nwalkers*(nsteps-iburn), nxparam)

            # Construct attrs dictionary
            adic = {"iburn" : iburn}

            # Delete existing dataset if it already exists
            if newname in f["mcmc/"].keys():
                del f["mcmc/"+newname]

            # Flatten chains
            if verbose: print("Flattening chains beyond burn-in (slow, especially if low burn-in index)...")
            flat_samples = samples[:,iburn:,:].reshape((-1, nparam))

            # Different approach if there are regularization params
            if (n_regparam > 0):
                if verbose: print("Filling xsample dataset...")
                sys.stdout.flush()
                # Loop over walkers
                # Exclude regularization parameters from albedo, area samples
                xsam = np.array([reparameterize.transform_Y2X(flat_samples[i,:-1*n_regparam],
                                n_type, n_band, n_slice, flatten=True)
                                for i in range(len(flat_samples))]
                                )
            else:
                # Use all parameters
                xsam = np.array([reparameterize.transform_Y2X(flat_samples[i], n_type, n_band,
                                n_slice, flatten=True)
                                for i in range(len(flat_samples))]
                                )

            # Create new dataset in existing hdf5 file
            xs = f.create_dataset("mcmc/"+newname, data=xsam, compression='lzf')
            # Add attributes to dataset
            for key, value in adic.iteritems(): xs.attrs[key] = value

    def _plot_posteriors(self, xsname="physical_samples", newdir="physical_posteriors/",
                         which=None, verbose=True):
        """Plot physical posteriors

        Parameters
        ----------
        xsname : str, optional
            Name of physical samples dataset in hdf5 file
        newdir : str, optional
            Name of new directory to hold posterior plots
        which : int, optional
            Index of parameter to plot individually, default `None` will plot all
        verbose : bool, optional
            Set to print status updates
        """
        f = self.hfile
        mdir = os.path.split(self.hpath)[0]
        # Create directory for plots
        ndir = os.path.join(mdir, newdir)
        try:
            os.mkdir(ndir)
            if verbose: print("Created directory:", ndir)
        except OSError:
            if verbose: print(ndir, "already exists.")
        # Are xsamples in the hdf5 file already?
        if xsname in f["mcmc/"].keys():
            # load physical samples
            xs = f["mcmc/"+xsname]
            # Plot posteriors
            plot_posteriors(xs, X_names=f["mcmc"].attrs["X_names"], directory=ndir, which=which)
        else:
            print("Error: '%s' not in hdf5 file! Try running transform_samples first." %xsname)


    def _plot_area_alb(self, xsname="physical_samples", epoxi=False):
        """Plot area covering fractions and albedos

        Parameters
        ----------
        xsname : str, optional
            Name of physical samples dataset in hdf5 file
        """
        # Define quantile intervals (1-sigma)
        intvls=[0.16, 0.5, 0.84]
        # Simplifying assignments
        f = self.hfile
        mdir = os.path.split(self.hpath)[0]
        # Are xsamples in the hdf5 file already?
        if xsname in f["mcmc/"].keys():
            # load physical samples
            xs = f["mcmc/"+xsname]
        else:
            print("Error: '%s' not in hdf5 file! Try running transform_samples first." %xsname)
            return
        # Plot and save
        quantiles = plot_area_alb(xs, self.N, directory=mdir, savetxt=True,
                                  intvls=intvls, epoxi=epoxi, eyecolors=False,
                                  lam=f["data/wlc_i"].value)
        # Delete save, if already exists
        if "quantiles" in f["mcmc/"].keys():
            del f["mcmc/quantiles"]
            print("Overwritting 'mcmc/quantiles' in hdf5 file")
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

    def _plot_model_data(self, iburn, newdir="model_data_compare/", verbose=True):
        """Plot the model vs data. Plots will be saved in `newdir` directory.

        Parameters
        ----------
        iburn : int
            Burn-in index
        newdir : str, optional
            Name of new directory that will hold model-data plots
        verbose : bool, optional
            Set to print status updates
        """
        f = self.hfile
        model_ij = f["mcmc/model_ij"]
        Obs_ij = f["data/Obs_ij"]
        Obsnoise_ij = f["data/Obsnoise_ij"]
        mdir = os.path.split(self.hpath)[0]
        # Create directory for plots
        ndir = os.path.join(mdir, newdir)
        try:
            os.mkdir(ndir)
            if verbose: print("Created directory:", ndir)
        except OSError:
            if verbose: print(ndir, "already exists.")
        # Plot model vs data
        plot_model_data(model_ij, Obs_ij, Obsnoise_ij, self.N, iburn=iburn,
                        directory=ndir)
################################################################################
# Data
################################################################################

class Data(object):
    """
    """
    def __init__(self, Time_i=None, Obs_ij=None, Obsnoise_ij=None, wlc_i=None, wlw_i=None,
                 lat_s=None, lon_s=None, lat_o=None, lon_o=None, period=None):
        """
        ``samurai`` Data object

        Parameters
        ----------
        Time_i : numpy.ndarray
            Observational time grid [hours]
        Obs_ij : numpy.ndarray
            Observed multi-wavelength, lightcurve data
        Obsnoise_ij : numpy.ndarray
            Observational errors
        wlc_i : numpy.ndarray
            Wavelength grid band centers [nm]
        wlw_i : numpy.ndarray
            Wavelength grid bandwidths [nm]
        lat_s : float
            Sub-stellar latitude [deg]
        lon_s : float
            Sub-stellar longitude [deg]
        lat_o : float
            sub-observer latitude [deg]
        lon_o : float
            Sub-observer longitude [deg]
        period : float
            Planet rotational period [hours]
        """
        self.Time_i=Time_i
        self.Obs_ij=Obs_ij
        self.Obsnoise_ij=Obsnoise_ij
        self.wlc_i=wlc_i
        self.wlw_i=wlw_i
        self.lat_s=lat_s
        self.lon_s=lon_s
        self.lat_o=lat_o
        self.lon_o=lon_o
        self._period=period
        if self._period is None:
            self._omega = None
        else:
            self._omega= ( 2. * np.pi / self.period )

    def get_dict(self):
        d = {}
        for key, value in self.__dict__.iteritems():
            if key.startswith("_"):
                nkey = key[1:]
            else:
                nkey = key
            d[nkey] = value
        return d

    @property
    def period(self):
        return self._period

    @period.setter
    def period(self, value):
        self._period = value
        self._omega = ( 2. * np.pi / value )

    @property
    def omega(self):
        return self._omega

    @omega.setter
    def omega(self, value):
        self._omega = value

    #

    @classmethod
    def from_EPOXI_march(cls):
        """
        Initialize Data using March 2008 EPOXI observations
        """
        infile = "../data/raddata_1_norm"
        period = 24.0
        lat_s = -0.581   # sub-solar latitude
        lon_s = 262.909  # sub-solar longitude
        lat_o = 1.678    # sub-observer latitude
        lon_o = 205.423  # sub-observer longitude
        Time_i = np.arange(25)*1.
        Obs_ij = np.loadtxt(os.path.join(RELPATH, infile))
        wlc_i = np.array([350., 450., 550., 650., 750., 850., 950.])
        wlw_i = np.array([100., 100., 100., 100., 100., 100., 100.])
        Obsnoise_ij = None
        # Return new class instance
        return cls(Time_i=Time_i, Obs_ij=Obs_ij, Obsnoise_ij=Obsnoise_ij,
                   wlc_i=wlc_i, wlw_i=wlw_i, lat_s=lat_s, lon_s=lon_s, lat_o=lat_o,
                   lon_o=lon_o, period=period)

    @classmethod
    def from_EPOXI_june(cls):
        """
        Initialize Data using June 2008 EPOXI observations
        """
        infile = "../data/raddata_2_norm"
        period = 24.0
        lat_s = 22.531  # sub-solar latitude
        lon_s = 280.977 # sub-solar longitude
        lat_o = 0.264   # sub-observer latitude
        lon_o = 205.465 # sub-observer longitude
        Time_i = np.arange(25)*1.
        Obs_ij = np.loadtxt(os.path.join(RELPATH, infile))
        wlc_i = np.array([350., 450., 550., 650., 750., 850., 950.])
        wlw_i = np.array([100., 100., 100., 100., 100., 100., 100.])
        Obsnoise_ij = None
        # Return new class instance
        return cls(Time_i=Time_i, Obs_ij=Obs_ij, Obsnoise_ij=Obsnoise_ij,
                   wlc_i=wlc_i, wlw_i=wlw_i, lat_s=lat_s, lon_s=lon_s, lat_o=lat_o,
                   lon_o=lon_o, period=period)

    @classmethod
    def from_test_simpleIGBP(cls):
        """
        Initialize Data using the simple IGBP map
        """
        infile = '../data/simpleIGBP_quadrature_lc'
        period = 7.0
        lat_s = 0.0  # sub-solar latitude
        lon_s = 90.0 # sub-solar longitude
        lat_o = 0.0  # sub-observer latitude
        lon_o = 0.0  # sub-observer longitude
        Time_i = np.arange(7)/7.*24.
        Obs_ij = np.loadtxt(os.path.join(RELPATH, infile))
        wlc_i = np.array([550., 650., 750., 850.])
        wlw_i = np.array([100., 100., 100., 100.])
        Obsnoise_ij = None
        # Return new class instance
        return cls(Time_i=Time_i, Obs_ij=Obs_ij, Obsnoise_ij=Obsnoise_ij,
                   wlc_i=wlc_i, wlw_i=wlw_i, lat_s=lat_s, lon_s=lon_s, lat_o=lat_o,
                   lon_o=lon_o, period=period)

################################################################################
# Mapper
################################################################################

class Mapper(object):
    """
    A mapping object and interface to the Surface Albedo Mapping Using RotAtional
    Inversion (``samurai``) model.

    Parameters
    ----------
    fmodel : str
        Forward model ("map" or "lightcurve")
    imodel : str
        Inverse model ("emcee" or "emcee3")
    data : samurai.Data
        Data object
    ntype : int
        Number of surface types
    nsideseed : int
        ``nside = 2 * 2 ** nsideseed``
    regularization : str
        Type of regularization term1
    reg_area : bool
    reg_alnd : bool
    sigmay : float
    noiselevel : float
    Nmcmc : int
        Number of MCMC iterations
    Nmcmc_b : int
        Number of MCMC burn-in iterations
    mcmc_seedamp : float
        Amplitude of gaussian ball for starting state
    hdf5_compression : str
        Compression algorithm for HDF5 datasets
    nslice : int
        Number of longitudinal slices in map ``fmodel``
    ncpu : int
        Number of CPUs to use for multithreading MCMC
    output : samurai.Output
        Object containing location and opened HDF5 file
    use_grey : bool
        Include a grey (spectrally uniform) contribution in the model
        (but not spatially uniform)
    use_global : bool
        Include a global (spatially uniform) contribution  in the model
        (but not spectrally uniform)
    max_dev : float
        Maximum deviation from flat line allowed for grey albedo spectrum and
        homogenous global coverage
    """
    def __init__(self, fmodel="map", imodel="emcee", data=None,
                 ntype=3, nsideseed=4, regularization=None, reg_area=False, reg_albd=False,
                 sigmay=3.0, noiselevel=0.01, Nmcmc=10000, Nmcmc_b=0, mcmc_seedamp=0.5,
                 hdf5_compression='lzf', nslice=9, ncpu=None, output=None,
                 use_grey = False, use_global = False
                 ):
        self.fmodel=fmodel
        self.imodel=imodel
        self.data=data
        self.ntype=ntype
        self.nsideseed=nsideseed
        self.regularization=regularization
        self.reg_area=reg_area
        self.reg_albd=reg_albd
        self.sigmay=sigmay
        self.noiselevel=noiselevel
        self.Nmcmc=Nmcmc
        self.Nmcmc_b=Nmcmc_b
        self.mcmc_seedamp=mcmc_seedamp
        self.hdf5_compression=hdf5_compression
        self.ncpu=None
        if self.ncpu is None:
            self.ncpu = multiprocessing.cpu_count()
        self.output=output
        self.use_grey = use_grey
        self.use_global = use_global

        # Params unique to map model
        self.nslice=nslice
        self._regularization=regularization
        self._nregparam=set_nreg(self.regularization)

    def get_dict(self):
        skip_list = ["data", "output"]
        d = {}
        for key, value in self.__dict__.iteritems():
            if key.startswith("_"):
                nkey = key[1:]
            else:
                nkey = key
            if key not in skip_list:
                d[nkey] = value
        return d

    #

    @property
    def regularization(self):
        return self._regularization

    @regularization.setter
    def regularization(self, value):
        self._regularization = value
        self._nregparam = set_nreg(value)

    @property
    def nregparam(self):
        return self._nregparam

    @nregparam.setter
    def nregparam(self, value):
        self._nregparam = value

    @classmethod
    def from_hdf5(cls, path):
        """
        Initialize Mapper object using previously saved HDF5 file

        Parameters
        ----------
        path : str
            Path to ``samurai`` output HDF5 file

        Returns
        -------
        Mapper object
        """

        # List of items to skip in initialization
        skip_list = ["nregparam", "omega", "Kernel_il"]

        # Load MCMC samples
        try:
            # Open the file stream
            f = h5py.File(path, 'r')
        except IOError:
            print("Error: HDF5 does not exist as suggested")
            return None

        # Create dictonary of Mapper attrs
        sdic = {}
        for key, value in f.attrs.iteritems():
            v = value
            if hasattr(value, "__len__"):
                if len(value) == 0:
                    v = None
            if key not in skip_list:
                sdic[key] = v

        # Create dictionary of Data
        ddic = {}
        for key, value in f["data"].attrs.iteritems():
            v = value
            if hasattr(value, "__len__"):
                if len(value) == 0:
                    v = None
            if key not in skip_list:
                ddic[key] = v
        for key, value in f["data"].iteritems():
            v = value.value
            if hasattr(value, "__len__"):
                if len(value) == 0:
                    v = None
            if key not in skip_list:
                ddic[key] = v

        # Return new class instance
        return cls(output=Output(hpath=path), data=Data(**ddic), **sdic)

    def run_oe(self, savedir="mcmc_output", tag=None, verbose=False, N=1):
        """
        Run Mapper object simulation using Optimal Estimation (OE)

        Parameters
        ----------
        """

        # Get start time
        now = datetime.datetime.now()
        if verbose: print(now.strftime("%Y-%m-%d %H:%M:%S"))

        # Create directory for this run
        if tag is None:
            startstr = now.strftime("%Y-%m-%d--%H-%M")
        else:
            startstr = tag

        # Create savedir directory, if necessary
        if savedir is not None:
            run_dir = os.path.join(savedir, startstr)
            try:
                os.mkdir(savedir)
                if verbose: print("Created directory:", savedir)
            except OSError:
                if verbose: print(savedir, "already exists.")
        else:
            run_dir = os.path.join("", startstr)
        # Create unique run_dir directory
        os.mkdir(run_dir)
        if verbose: print("Created directory:", run_dir)

        # Unpack class variables
        fmodel = self.fmodel
        imodel = self.imodel
        Time_i = self.data.Time_i
        ntype = self.ntype
        nregparam = self.nregparam
        regularization = self.regularization
        lat_o = self.data.lat_o
        lon_o = self.data.lon_o
        lat_s = self.data.lat_s
        lon_s = self.data.lon_s
        omega = self.data.omega
        ncpu = self.ncpu
        num_mcmc = self.Nmcmc
        seed_amp  = self.mcmc_seedamp
        hdf5_compression = self.hdf5_compression
        waveband_centers = self.data.wlc_i
        waveband_widths = self.data.wlw_i
        use_grey = self.use_grey
        use_global = self.use_global

        # Input data
        Obs_ij = self.data.Obs_ij
        if self.data.Obsnoise_ij is None:
            Obsnoise_ij = ( self.noiselevel * self.data.Obs_ij )
        else:
            Obsnoise_ij = self.data.Obsnoise_ij

        nband = len(Obs_ij[0])

        # Calculate n_side
        nside = 2*2**self.nsideseed

        # Set geometric kernel depending on model
        if fmodel == "map":
            nslice = self.nslice
            param_geometry = ( lat_o, lon_o, lat_s, lon_s, omega )
            Kernel_il = geometry.kernel( Time_i, nslice, nside, param_geometry )
        elif fmodel == "lightcurve":
            nslice = len(Obs_ij)
            Kernel_il = np.identity( nslice )
        else:
            print("Error: %s is an unrecognized forward model" %fmodel)
            return None

        # Create list of strings for Y & X parameter names
        Y_names, X_names = generate_tex_names(ntype, nband, nslice)

        # Specify hdf5 save file and group names
        hfile = os.path.join(run_dir, "samurai_out.hdf5")
        grp_init_name = "oe"
        grp_mcmc_name = "mcmc"
        grp_data_name = "data"
        compression = hdf5_compression

        # Get object attributes for saving to HDF5
        hfile_attrs = self.get_dict()
        tmp_dict = self.data.get_dict()
        data_dict_datasets = {}
        data_dict_attrs = {}
        data_dict_datasets["Kernel_il"] = Kernel_il # Add Kernel dict
        # Partition data values into attributes and datasets
        for key, value in tmp_dict.iteritems():
            if hasattr(value, "__len__"):
                data_dict_datasets[key] = value
            else:
                data_dict_attrs[key] = value

        # print
        if verbose: print("Saving:", hfile)

        mcmc_dict_attrs = {
            "Y_names" : Y_names,
            "X_names" : X_names,
        }

        # Create hdf5 file
        f = h5py.File(hfile, 'w')

        # Add global metadata
        for key, value in hfile_attrs.iteritems():
            if value is None:
                f.attrs[key] = ()
            else:
                f.attrs[key] = value

        # Create hdf5 groups (like a directory structure)
        grp_init = f.create_group(grp_init_name)    # f["initial_optimization/"]
        grp_data = f.create_group(grp_data_name)    # f["data/"]

        # Save data datasets
        for key, value in data_dict_datasets.iteritems():
            if value is None:
                grp_data.create_dataset(key, data=(), compression=compression)
            else:
                grp_data.create_dataset(key, data=value, compression=compression)
        # Save data metadata
        for key, value in data_dict_attrs.iteritems():
            if value is None:
                grp_data.attrs[key] = ()
            else:
                grp_data.attrs[key] = value


        for i in range(N):

            # Initialize the fitting parameters
            #X0_albd_kj = 0.3+np.zeros([ntype, nband])
            #X0_area_lk = 0.2+np.zeros([nslice, ntype])

            # Randomize inital fitting parameters
            X0_albd_kj = np.random.rand(ntype, nband)
            X0_area_lk = np.zeros([nslice, ntype]) # intialize array
            for il in range(nslice):
                tmp = np.random.rand(ntype-1)
                while np.sum(tmp) > 1.0:
                    tmp = np.random.rand(ntype-1)
                last = 1. - np.sum(tmp)
                X0_area_lk[il,:] = np.hstack([tmp,last])

            Y0_array = reparameterize.transform_X2Y(X0_albd_kj, X0_area_lk)
            if ( nregparam > 0 ) :
                Y0_array = np.append(Y0_array, np.array([10.]*nregparam) )
            n_dim = len(Y0_array)
            if verbose:
                print('Y0_array', Y0_array)
                print('# of parameters', n_dim)
                print('N_REGPARAM', nregparam)
            if (nregparam > 0):
                X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(Y0_array[:-1*nregparam], ntype, nband, nslice)
            else:
                X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(Y0_array, ntype, nband, nslice)

            ############ run minimization ############

            # minimize
            if verbose: print("finding best-fit values...")
            data = (Obs_ij, Obsnoise_ij, Kernel_il, regularization, nregparam, True, False, ntype, nslice)
            #output = minimize(lnprob, Y0_array, args=data, method="Nelder-Mead")
            output = minimize(lnprob, Y0_array, args=data, method="L-BFGS-B" )
            best_fit = output["x"]
            if verbose: print("best-fit", best_fit)

            # more information about the best-fit parameters
            data = (Obs_ij, Obsnoise_ij, Kernel_il, regularization, nregparam, True, False, ntype, nslice)
            lnprob_bestfit = lnprob( output['x'], *data )

            # compute BIC
            BIC = 2.0 * lnprob_bestfit + len( output['x'] ) * np.log( len(Obs_ij.flatten()) )
            if verbose: print('BIC: ', BIC)

            # best-fit values for physical parameters
            if nregparam > 0:
                X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(output["x"][:-1*nregparam], ntype, nband, nslice)
            else :
                X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(output["x"], ntype, nband, nslice)

            X_albd_kj_T = X_albd_kj.T

            # best-fit values for regularizing parameters
            if regularization is not None:
                if regularization == 'Tikhonov' :
                    if verbose: print('sigma', best_fit[-1])
                elif regularization == 'GP' :
                    if verbose: print('overall_amp', best_fit[-3])
                    if verbose: print('wn_rel_amp', np.exp( best_fit[-2] ) / ( 1. + np.exp( best_fit[-2] ) ))
                    if verbose: print('lambda _angular', best_fit[-1] * ( 180. / np.pi ))
                elif regularization == 'GP2' :
                    if verbose: print('overall_amp', best_fit[-2])
                    if verbose: print('lambda _angular', best_fit[-1]* ( 180. / np.pi ))

            # Flatten best-fitting physical parameters
            bestfit = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

            # Create dictionaries of initial results to convert to hdf5
            # datasets and attributes
            init_dict_datasets = {
                "X0_albd_kj" : X0_albd_kj,
                "X0_area_lk" : X0_area_lk,
                "best_fity" : best_fit,
                "X_area_lk" : X_area_lk,
                "X_albd_kj_T" : X_albd_kj_T,
                "best_fitx" : bestfit
            }
            init_dict_attrs = {
                "best_lnprob" : lnprob_bestfit,
                "best_BIC" : BIC
            }

            grp_index = grp_init.create_group(str(i))

            # Save initial run datasets
            for key, value in init_dict_datasets.iteritems():
                if value is None:
                    grp_index.create_dataset(key, data=(), compression=compression)
                else:
                    grp_index.create_dataset(key, data=value, compression=compression)
            # Save initial run metadata
            for key, value in init_dict_attrs.iteritems():
                if value is None:
                    grp_index.attrs[key] = ()
                else:
                    grp_index.attrs[key] = value

        # Close hdf5 file stream
        f.close()

        # Save path to HDF5 file in output object
        self.output = Output(hpath=hfile)

    def run_oe_atmosphere(self, savedir="mcmc_output", tag=None, verbose=False, N=1):
        """
        Run Mapper object simulation using Optimal Estimation (OE)

        Parameters
        ----------
        """

        # Get start time
        now = datetime.datetime.now()
        if verbose: print(now.strftime("%Y-%m-%d %H:%M:%S"))

        # Create directory for this run
        if tag is None:
            startstr = now.strftime("%Y-%m-%d--%H-%M")
        else:
            startstr = tag

        # Create savedir directory, if necessary
        if savedir is not None:
            run_dir = os.path.join(savedir, startstr)
            try:
                os.mkdir(savedir)
                if verbose: print("Created directory:", savedir)
            except OSError:
                if verbose: print(savedir, "already exists.")
        else:
            run_dir = os.path.join("", startstr)

        # Create unique run_dir directory
        try:
            os.mkdir(run_dir)
            if verbose: print("Created directory:", run_dir)
        except OSError:
            if verbose: print(run_dir, "already exists.")

        # Unpack class variables
        fmodel = self.fmodel
        imodel = self.imodel
        Time_i = self.data.Time_i
        ntype = self.ntype
        nregparam = self.nregparam
        regularization = self.regularization
        lat_o = self.data.lat_o
        lon_o = self.data.lon_o
        lat_s = self.data.lat_s
        lon_s = self.data.lon_s
        omega = self.data.omega
        ncpu = self.ncpu
        num_mcmc = self.Nmcmc
        seed_amp  = self.mcmc_seedamp
        hdf5_compression = self.hdf5_compression
        waveband_centers = self.data.wlc_i
        waveband_widths = self.data.wlw_i
        use_grey = self.use_grey
        use_global = self.use_global

        # Input data
        Obs_ij = self.data.Obs_ij
        if self.data.Obsnoise_ij is None:
            Obsnoise_ij = ( self.noiselevel * self.data.Obs_ij )
        else:
            Obsnoise_ij = self.data.Obsnoise_ij

        nband = len(Obs_ij[0])

        # Calculate n_side
        nside = 2*2**self.nsideseed

        # Set geometric kernel depending on model
        if fmodel == "map":
            nslice = self.nslice
            param_geometry = ( lat_o, lon_o, lat_s, lon_s, omega )
            Kernel_il = geometry.kernel( Time_i, nslice, nside, param_geometry )
        elif fmodel == "lightcurve":
            nslice = len(Obs_ij)
            Kernel_il = np.identity( nslice )
        else:
            print("Error: %s is an unrecognized forward model" %fmodel)
            return None

        # Create list of strings for Y & X parameter names
        Y_names, X_names = generate_tex_names(ntype, nband, nslice)

        # Specify hdf5 save file and group names
        hfile = os.path.join(run_dir, "samurai_out.hdf5")
        grp_init_name = "oe"
        grp_mcmc_name = "mcmc"
        grp_data_name = "data"
        compression = hdf5_compression

        # Get object attributes for saving to HDF5
        hfile_attrs = self.get_dict()
        tmp_dict = self.data.get_dict()
        data_dict_datasets = {}
        data_dict_attrs = {}
        data_dict_datasets["Kernel_il"] = Kernel_il # Add Kernel dict
        # Partition data values into attributes and datasets
        for key, value in tmp_dict.iteritems():
            if hasattr(value, "__len__"):
                data_dict_datasets[key] = value
            else:
                data_dict_attrs[key] = value

        # print
        if verbose: print("Saving:", hfile)

        mcmc_dict_attrs = {
            "Y_names" : Y_names,
            "X_names" : X_names,
        }

        # Create hdf5 file
        f = h5py.File(hfile, 'w')

        # Create output object
        self.output = Output(hpath=hfile)

        # Add global metadata
        for key, value in hfile_attrs.iteritems():
            if value is None:
                f.attrs[key] = ()
            else:
                f.attrs[key] = value

        # Create hdf5 groups (like a directory structure)
        grp_init = f.create_group(grp_init_name)    # f["initial_optimization/"]
        grp_data = f.create_group(grp_data_name)    # f["data/"]

        # Save data datasets
        for key, value in data_dict_datasets.iteritems():
            if value is None:
                grp_data.create_dataset(key, data=(), compression=compression)
            else:
                grp_data.create_dataset(key, data=value, compression=compression)
        # Save data metadata
        for key, value in data_dict_attrs.iteritems():
            if value is None:
                grp_data.attrs[key] = ()
            else:
                grp_data.attrs[key] = value


        for i in range(N):

            # Account for atmosphere
            if use_grey:
                # Add another surface type
                #ntype += 1
                # Sample initial area vector
                X_grey_area_l = np.random.rand(nslice)
                # Sample initial grey albedo
                X_grey_albd_0 = np.random.rand()
                # Construct flat spectrum
                X_grey_albd_j = X_grey_albd_0 * np.ones(nband)
            if use_global:
                # Add another surface type
                #ntype += 1
                # Sample initial albedo vector
                X_global_albd_j = np.random.rand(nband)
                # Sample initial global coverage
                X_global_area_0 = np.random.rand()
                # Construct flat longitude map
                X_global_area_l = X_global_area_0 * np.ones(nslice)

            # Randomize inital fitting parameters (uniform)
            if (not use_grey) and (not use_global):
                # Randomize inital fitting parameters without atmosphere
                X0_albd_kj = np.random.rand(ntype, nband)
                X0_area_lk = np.zeros([nslice, ntype]) # intialize array
                for il in range(nslice):
                    tmp = np.random.rand(ntype-1)
                    while np.sum(tmp) > 1.0:
                        tmp = np.random.rand(ntype-1)
                    last = 1. - np.sum(tmp)
                    X0_area_lk[il,:] = np.hstack([tmp,last])
            elif use_grey and (not use_global):
                # Randomize inital fitting parameters with grey component to atmosphere
                X0_albd_kj = np.random.rand(ntype, nband)
                # Insert grey albedo vector into last column
                X0_albd_kj[-1,:] = X_grey_albd_j
                # Initialize map
                X0_area_lk = np.zeros([nslice, ntype]) # intialize array
                for il in range(nslice):
                    tmp = np.random.rand(ntype-1)
                    while np.sum(tmp) > 1.0:
                        tmp = np.random.rand(ntype-1)
                    last = 1. - np.sum(tmp)
                    X0_area_lk[il,:] = np.hstack([tmp,last])
            elif use_global and (not use_grey):
                # Randomize inital fitting parameters with global component to atmosphere
                X0_albd_kj = np.random.rand(ntype, nband)
                # Initialize map
                X0_area_lk = np.zeros([nslice, ntype]) # intialize array
                for il in range(nslice):
                    tmp = np.random.rand(ntype-1)
                    tmp[-1] = X_global_area_0
                    while np.sum(tmp) > 1.0:
                        tmp = np.random.rand(ntype-1)
                        tmp[-1] = X_global_area_0
                    last = 1. - np.sum(tmp)
                    X0_area_lk[il,:] = np.hstack([last,tmp])
                pass
            else:
                # Randomize inital fitting parameters with global and grey
                # component atmosphere
                X0_albd_kj = np.random.rand(ntype, nband)
                # Insert grey albedo vector into last column
                X0_albd_kj[-2,:] = X_grey_albd_j # grey is second to last
                # Initialize map
                X0_area_lk = np.zeros([nslice, ntype]) # intialize array
                for il in range(nslice):
                    tmp = np.random.rand(ntype-1)
                    tmp[-1] = X_global_area_0    # global is last
                    while np.sum(tmp) > 1.0:
                        tmp = np.random.rand(ntype-1)
                        tmp[-1] = X_global_area_0
                    last = 1. - np.sum(tmp)
                    X0_area_lk[il,:] = np.hstack([last,tmp])

            # Apply reparameterization:
            # Physical albedos and areas (X) --> Parameterized state vector
            Y0_array = reparameterize.transform_X2Y_atmosphere(X0_albd_kj, X0_area_lk, use_grey=use_grey, use_global=use_global)
            # If using regularization
            if ( nregparam > 0 ) :
                Y0_array = np.append(Y0_array, np.array([10.]*nregparam) )

            # Calculate dimensionality of parameter space
            n_dim = len(Y0_array)

            # Print diagnostics?
            if verbose:
                print('Y0_array', Y0_array)
                print('# of parameters', n_dim)
                print('N_REGPARAM', nregparam)

            # Test reverse (de?)parameterization:
            # Y --> X
            if (nregparam > 0):
                # If using any regularization
                X_albd_kj, X_area_lk = reparameterize.transform_Y2X_atmosphere(Y0_array[:-1*nregparam], ntype, nband, nslice, use_grey = use_grey, use_global = use_global)
            else:
                # If no regularization
                X_albd_kj, X_area_lk = reparameterize.transform_Y2X_atmosphere(Y0_array, ntype, nband, nslice, use_grey = use_grey, use_global = use_global)

            ############ run minimization ############

            # minimize
            if verbose: print("finding best-fit values...")
            data = (Obs_ij, Obsnoise_ij, Kernel_il, regularization, nregparam, True, False, ntype, nslice, use_grey, use_global)
            #output = minimize(lnprob, Y0_array, args=data, method="Nelder-Mead")
            output = minimize(lnprob_atmosphere, Y0_array, args=data, method="L-BFGS-B" )
            best_fit = output["x"]
            if verbose: print("best-fit", best_fit)

            # more information about the best-fit parameters
            data = (Obs_ij, Obsnoise_ij, Kernel_il, regularization, nregparam, True, False, ntype, nslice, use_grey, use_global)
            lnprob_bestfit = lnprob_atmosphere( output['x'], *data )

            # compute BIC
            BIC = 2.0 * lnprob_bestfit + len( output['x'] ) * np.log( len(Obs_ij.flatten()) )
            if verbose: print('BIC: ', BIC)

            # best-fit values for physical parameters
            if nregparam > 0:
                X_albd_kj, X_area_lk =  reparameterize.transform_Y2X_atmosphere(output["x"][:-1*nregparam], ntype, nband, nslice, use_grey = use_grey, use_global = use_global)
            else :
                X_albd_kj, X_area_lk =  reparameterize.transform_Y2X_atmosphere(output["x"], ntype, nband, nslice, use_grey = use_grey, use_global = use_global)

            X_albd_kj_T = X_albd_kj.T

            # best-fit values for regularizing parameters
            if regularization is not None:
                if regularization == 'Tikhonov' :
                    if verbose: print('sigma', best_fit[-1])
                elif regularization == 'GP' :
                    if verbose: print('overall_amp', best_fit[-3])
                    if verbose: print('wn_rel_amp', np.exp( best_fit[-2] ) / ( 1. + np.exp( best_fit[-2] ) ))
                    if verbose: print('lambda _angular', best_fit[-1] * ( 180. / np.pi ))
                elif regularization == 'GP2' :
                    if verbose: print('overall_amp', best_fit[-2])
                    if verbose: print('lambda _angular', best_fit[-1]* ( 180. / np.pi ))

            # Flatten best-fitting physical parameters
            bestfit = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

            # Create dictionaries of initial results to convert to hdf5
            # datasets and attributes
            init_dict_datasets = {
                "X0_albd_kj" : X0_albd_kj,
                "X0_area_lk" : X0_area_lk,
                "best_fity" : best_fit,
                "X_area_lk" : X_area_lk,
                "X_albd_kj_T" : X_albd_kj_T,
                "best_fitx" : bestfit
            }
            init_dict_attrs = {
                "best_lnprob" : lnprob_bestfit,
                "best_BIC" : BIC
            }

            grp_index = grp_init.create_group(str(i))

            # Save initial run datasets
            for key, value in init_dict_datasets.iteritems():
                if value is None:
                    grp_index.create_dataset(key, data=(), compression=compression)
                else:
                    grp_index.create_dataset(key, data=value, compression=compression)
            # Save initial run metadata
            for key, value in init_dict_attrs.iteritems():
                if value is None:
                    grp_index.attrs[key] = ()
                else:
                    grp_index.attrs[key] = value

        # Close hdf5 file stream
        f.close()

    def run_mcmc(self, savedir="mcmc_output", tag=None, verbose=True,
                 resume=False, initial_guess=None):
        """
        Run Mapper object simulation

        Parameters
        ----------
        savedir : str, optional
            Relative directory to save output files
        tag : str, optional
            Specify saving tag for output (default is the time)
        verbose : bool, optional
            Set to print status updates
        resume : bool, optional
            Set to resume MCMC from last saved state
        initial_guess : array-like, optional
            Initial guess around which mcmc chains will be initialized
        """

        if not resume:

            # Get start time
            now = datetime.datetime.now()
            if verbose: print(now.strftime("%Y-%m-%d %H:%M:%S"))

            # Create directory for this run
            if tag is None:
                startstr = now.strftime("%Y-%m-%d--%H-%M")
            else:
                startstr = tag

            # Create savedir directory, if necessary
            if savedir is not None:
                run_dir = os.path.join(savedir, startstr)
                try:
                    os.mkdir(savedir)
                    if verbose: print("Created directory:", savedir)
                except OSError:
                    if verbose: print(savedir, "already exists.")
            else:
                run_dir = os.path.join("", startstr)
            # Create unique run_dir directory
            os.mkdir(run_dir)
            if verbose: print("Created directory:", run_dir)
        else:
            # Set run_dir by previous run
            run_dir = os.path.split(self.output.hpath)[0]

        # Unpack class variables
        fmodel = self.fmodel
        imodel = self.imodel
        Time_i = self.data.Time_i
        ntype = self.ntype
        nregparam = self.nregparam
        regularization = self.regularization
        lat_o = self.data.lat_o
        lon_o = self.data.lon_o
        lat_s = self.data.lat_s
        lon_s = self.data.lon_s
        omega = self.data.omega
        ncpu = self.ncpu
        num_mcmc = self.Nmcmc
        seed_amp  = self.mcmc_seedamp
        hdf5_compression = self.hdf5_compression
        waveband_centers = self.data.wlc_i
        waveband_widths = self.data.wlw_i

        # Input data
        Obs_ij = self.data.Obs_ij
        if self.data.Obsnoise_ij is None:
            Obsnoise_ij = ( self.noiselevel * self.data.Obs_ij )
        else:
            Obsnoise_ij = self.data.Obsnoise_ij

        nband = len(Obs_ij[0])

        # Calculate n_side
        nside = 2*2**self.nsideseed

        # Set geometric kernel depending on model
        if fmodel == "map":
            nslice = self.nslice
            param_geometry = ( lat_o, lon_o, lat_s, lon_s, omega )
            Kernel_il = geometry.kernel( Time_i, nslice, nside, param_geometry )
        elif fmodel == "lightcurve":
            nslice = len(Obs_ij)
            Kernel_il = np.identity( nslice )
        else:
            print("Error: %s is an unrecognized forward model" %fmodel)
            return None


        # Initialize the fitting parameters
        X0_albd_kj = 0.3+np.zeros([ntype, nband])
        X0_area_lk = 0.2+np.zeros([nslice, ntype])
        Y0_array = reparameterize.transform_X2Y(X0_albd_kj, X0_area_lk)
        if ( nregparam > 0 ) :
            Y0_array = np.append(Y0_array, np.array([10.]*nregparam) )
        n_dim = len(Y0_array)
        if verbose:
            print('Y0_array', Y0_array)
            print('# of parameters', n_dim)
            print('N_REGPARAM', nregparam)
        if (nregparam > 0):
            X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(Y0_array[:-1*nregparam], ntype, nband, nslice)
        else:
            X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(Y0_array, ntype, nband, nslice)

        # Create list of strings for Y & X parameter names
        Y_names, X_names = generate_tex_names(ntype, nband, nslice)

        if not resume:

            ############ run minimization ############

            # minimize
            if verbose: print("finding best-fit values...")
            data = (Obs_ij, Obsnoise_ij, Kernel_il, regularization, nregparam, True, False, ntype, nslice)
            output = minimize(lnprob, Y0_array, args=data, method="Nelder-Mead")
            #output = minimize(lnprob, Y0_array, args=data, method="L-BFGS-B" )
            best_fit = output["x"]
            if verbose: print("best-fit", best_fit)

            # more information about the best-fit parameters
            data = (Obs_ij, Obsnoise_ij, Kernel_il, regularization, nregparam, True, False, ntype, nslice)
            lnprob_bestfit = lnprob( output['x'], *data )

            # compute BIC
            BIC = 2.0 * lnprob_bestfit + len( output['x'] ) * np.log( len(Obs_ij.flatten()) )
            if verbose: print('BIC: ', BIC)

            # best-fit values for physical parameters
            if nregparam > 0:
                X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(output["x"][:-1*nregparam], ntype, nband, nslice)
            else :
                X_albd_kj, X_area_lk =  reparameterize.transform_Y2X(output["x"], ntype, nband, nslice)

            X_albd_kj_T = X_albd_kj.T

            # best-fit values for regularizing parameters
            if regularization is not None:
                if regularization == 'Tikhonov' :
                    if verbose: print('sigma', best_fit[-1])
                elif regularization == 'GP' :
                    if verbose: print('overall_amp', best_fit[-3])
                    if verbose: print('wn_rel_amp', np.exp( best_fit[-2] ) / ( 1. + np.exp( best_fit[-2] ) ))
                    if verbose: print('lambda _angular', best_fit[-1] * ( 180. / np.pi ))
                elif regularization == 'GP2' :
                    if verbose: print('overall_amp', best_fit[-2])
                    if verbose: print('lambda _angular', best_fit[-1]* ( 180. / np.pi ))

            # Flatten best-fitting physical parameters
            bestfit = np.r_[ X_albd_kj.flatten(), X_area_lk.T.flatten() ]

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

        ############ run inverse model ############

        # Define MCMC parameters
        n_dim = len(Y0_array)
        n_walkers = calculate_walkers(n_dim)

        # Define data tuple for emcee
        data = (Obs_ij, Obsnoise_ij, Kernel_il, regularization, nregparam, False, False, ntype, nslice)

        if resume:
            # Initialize chain/walker positions from previous state
            self.output.open()
            p0 = self.output.hfile["mcmc/samples"][:,-1,:]
            self.output.close()
        elif initial_guess is not None:
            # Initialize chain/walker state from Gaussian ball
            p0 = initial_guess + seed_amp * np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim))
        else:
            # Initialize chain/walker state from Gaussian ball
            p0 = best_fit + seed_amp * np.random.rand(n_dim * n_walkers).reshape((n_walkers, n_dim))

        if imodel == "emcee":

            # Initialize emcee EnsembleSampler
            sampler = emcee.EnsembleSampler(n_walkers, n_dim, lnprob, args=data, threads=ncpu)

            # Do Burn-in run? No
            if verbose: print("Running emcee...")

            # Run MCMC
            sampler.run_mcmc( p0, num_mcmc )

            # Extract chain from sampler
            original_samples = sampler.chain

            #import pdb; pdb.set_trace()
            # Get the integrated autocorrelation time for each dimension
            #acors = sampler.get_autocorr_time()
            #acor_max = int(np.floor(np.min(acors)))

            # Get model evaluations
            blobs = sampler.blobs
            shape = (len(blobs), len(blobs[0]), len(blobs[0][0]), len(blobs[0][0][0]))
            model_ij = np.reshape(blobs, shape)

            # Create dictionaries for mcmc data and metadata
            mcmc_dict_datasets = {
                "samples" : original_samples,
                "model_ij" : model_ij,
                "p0" : p0
            }

        elif imodel == "emcee3":

            # Initialize simple model
            model = emcee3.SimpleModel(lnlike, lnprior, args=data)

            # Initialize ensemble sampler
            ensemble = emcee3.Ensemble(model, p0)

            # Initialize sampler with Moves: Linear combo of samplers -- essential
            # for multi-modal posteriors
            sampler = emcee3.Sampler([
                (emcee3.moves.DEMove(0.01), 0.5),
                (emcee3.moves.DESnookerMove(), 0.1),
                (emcee3.moves.StretchMove(), 0.1),
            ])

            # Do Burn-in run? No
            if verbose: print("Running emcee3...")

            # Run MCMC
            ensemble = sampler.run(ensemble, num_mcmc, progress=verbose)

            # Extract chains, reshape so emcee compatible
            original_samples = sampler.get_coords().reshape([n_walkers, num_mcmc, n_dim])

            #import pdb; pdb.set_trace()
            # Get the integrated autocorrelation time for each dimension
            #acors = sampler.get_integrated_autocorr_time()
            #acor_max = int(np.floor(np.min(acors)))

            #model_ij = None

            # Create dictionaries for mcmc data and metadata
            mcmc_dict_datasets = {
                "samples" : original_samples,
                #"model_ij" : model_ij,
                "p0" : p0
            }

        else:
            print("Error: %s is an unrecognized inverse model" %imodel)
            return None

        ############ Save HDF5 File ############

        # Specify hdf5 save file and group names
        hfile = os.path.join(run_dir, "samurai_out.hdf5")
        grp_init_name = "initial_optimization"
        grp_mcmc_name = "mcmc"
        grp_data_name = "data"
        compression = hdf5_compression

        if resume:
            # Open existing output hdf5
            self.output.open()
            f = self.output.hfile
            # Add new chains and starting point to hdf5 file
            for key, value in mcmc_dict_datasets.iteritems():
                # Delete previous mcmc output group
                f.__delitem__(os.path.join(grp_mcmc_name, key))
                if value is None:
                    f[grp_mcmc_name].create_dataset(key, data=(), compression=compression)
                else:
                    f[grp_mcmc_name].create_dataset(key, data=value, compression=compression)
            # Close existing output hdf5
            self.output.close()
        else:

            # Get object attributes for saving to HDF5
            hfile_attrs = self.get_dict()
            tmp_dict = self.data.get_dict()
            data_dict_datasets = {}
            data_dict_attrs = {}
            data_dict_datasets["Kernel_il"] = Kernel_il # Add Kernel dict
            # Partition data values into attributes and datasets
            for key, value in tmp_dict.iteritems():
                if hasattr(value, "__len__"):
                    data_dict_datasets[key] = value
                else:
                    data_dict_attrs[key] = value

            # print
            if verbose: print("Saving:", hfile)

            mcmc_dict_attrs = {
                "Y_names" : Y_names,
                "X_names" : X_names,
            }

            # Create hdf5 file
            f = h5py.File(hfile, 'w')

            # Add global metadata
            for key, value in hfile_attrs.iteritems():
                if value is None:
                    f.attrs[key] = ()
                else:
                    f.attrs[key] = value

            # Create hdf5 groups (like a directory structure)
            grp_init = f.create_group(grp_init_name)    # f["initial_optimization/"]
            grp_data = f.create_group(grp_data_name)    # f["data/"]
            grp_mcmc = f.create_group(grp_mcmc_name)    # f[mcmc/]

            # Save initial run datasets
            for key, value in init_dict_datasets.iteritems():
                if value is None:
                    grp_init.create_dataset(key, data=(), compression=compression)
                else:
                    grp_init.create_dataset(key, data=value, compression=compression)
            # Save initial run metadata
            for key, value in init_dict_attrs.iteritems():
                if value is None:
                    grp_init.attrs[key] = ()
                else:
                    grp_init.attrs[key] = value

            # Save data datasets
            for key, value in data_dict_datasets.iteritems():
                if value is None:
                    grp_data.create_dataset(key, data=(), compression=compression)
                else:
                    grp_data.create_dataset(key, data=value, compression=compression)
            # Save data metadata
            for key, value in data_dict_attrs.iteritems():
                if value is None:
                    grp_data.attrs[key] = ()
                else:
                    grp_data.attrs[key] = value

            # Save mcmc run datasets
            for key, value in mcmc_dict_datasets.iteritems():
                if value is None:
                    grp_mcmc.create_dataset(key, data=(), compression=compression)
                else:
                    grp_mcmc.create_dataset(key, data=value, compression=compression)
            # Save mcmc run metadata
            for key, value in mcmc_dict_attrs.iteritems():
                if value is None:
                    grp_mcmc.attrs[key] = ()
                else:
                    grp_mcmc.attrs[key] = value

            # Close hdf5 file stream
            f.close()

            # Save path to HDF5 file in output object
            self.output = Output(hpath=hfile)

        # End if
