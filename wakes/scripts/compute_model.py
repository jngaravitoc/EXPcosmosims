"""

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d, BSpline, splrep
import pyEXP
from matplotlib.colors import LogNorm
import makemodel
from inspect import getmembers, isfunction
import pickle

def profile_plot(rlist, ylist, labels, figname):
    nprofiles = len(rlist)
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4))
    for n in range(nprofiles):
        ax.loglog(rlist[n], ylist[n], label=labels[n])
    plt.savefig("density_profiles_{}.png".figname, bbox_inches='tight')
    plt.close()
    
def load_halos_names():
    with open('../../data/NICO_PICKLE/times.pickle', 'rb') as f:
        times =  pickle.load(f)
    with open('../../data/NICO_PICKLE/start_snap.pickle', 'rb') as f:
        init_snap =  pickle.load(f)
    with open('../../data/NICO_PICKLE/tpericenter.pickle', 'rb') as f:
        tpericenter = pickle.load(f)
    #tsim = times + tpericenter
    return list(times.keys()), times, tpericenter,  init_snap 

def make_halo_config(basis_id, numr, rmin, rmax, lmax, nmax, rmapping, 
                modelname, cachename):
    """
    Creates a configuration file required to build a basis model.

    Args:
    basis_id (str): The identity of the basis model.
    numr (int): The number of radial grid points.
    rmin (float): The minimum radius value.
    rmax (float): The maximum radius value.
    lmax (int): The maximum l value of the basis.
    nmax (int): The maximum n value of the basis.
    scale (float): Scaling factor for the basis.
    modelname (str, optional): Name of the model. Default is an empty string.
    cachename (str, optional): Name of the cache file. Default is '.slgrid_sph_cache'.

    Returns:
    str: A string representation of the configuration file.

    Raises:
    None
    """
    
    config = 'id: {:s}\n'.format(basis_id)
    config += 'parameters:\n'
    config += '  numr: {:d}\n'.format(numr)
    config += '  rmin: {:.7f}\n'.format(rmin)
    config += '  rmax: {:.3f}\n'.format(rmax)
    config += '  Lmax: {:d}\n'.format(lmax)
    config += '  nmax: {:d}\n'.format(nmax)
    config += '  rmapping: {:.3f}\n'.format(rmapping)
    config += '  modelname: {}\n'.format(modelname)
    config += '  cachename: {}\n'.format(cachename)
    return config

def compute_coefficients(basis, coefs, positions, masses, halo):
    halo_coef = basis.createFromArray(masses, positions, time=k)
    coefs.add(halo_coef)

def write_coeff_attrbs(coefname, nmax, lmax, rmapping, config):
    f = h5py.File(coefname, 'r+')
    f.attrs['config'] = config
    f.attrs['lmax'] = lmax
    f.attrs['nmax'] = nmax
    f.attrs['rmapping'] = rmapping
    f.close()

if __name__ == "__main__":
    DATAPATH = "../../data/"
    halos, times, tperi, init_snap = load_halos_names()
    mp = 402830.0 # Msun 

    
    ## Basis details:
    basisID = 'sphereSL'
    numr = 1000
    rmin = 1.0
    rmax = 150
    lmax = 5
    nmax = 10
    rmapping = 25.0
    modelname = "mwest_nfw.txt"
    cachename = "mwest_nfw.cache"
    

    # EXP units and parameters 
    Renc = rmax # Maximum radius to build the basis
    Rin = rmin
    Rbins = numr # Number of bins 
    rvals_lin = np.linspace(Rin, Renc, Rbins) # Choosing bins
    rvals_log = np.logspace(np.log10(Rin), np.log10(Renc), Rbins) # Choosing bins
    smooth = 0.8
    

    # Make model 
    Rnfw, Dnfw, Mnfw, Pnfw = makemodel.makemodel(makemodel.powerhalo, M=1,
                                                funcargs=([rmapping, 0, 1, 2]),rvals=rvals_lin,
                                                pfile=modelname)




    halo_config = make_halo_config(basisID, numr, rmin, rmax, lmax, nmax, rmapping, modelname, cachename)
    halo_basis = pyEXP.basis.Basis.factory(halo_config)

    for halo in halos:
        print("-> Analyzing {}".format(halo))
        halo_times = times[halo]
        tperi_halo = tperi[halo]
        init_halo_snap = init_snap[halo]
        tsim = halo_times + tperi_halo
        Nsnaps = len(halo_times)
        print('Nsnaps:', Nsnaps)
        Memp_all = np.zeros((Nsnaps, len(rvals_lin)-1))
        Pemp_all = np.zeros_like(Memp_all)
        Remp_all = np.zeros_like(Memp_all)
        Demp_all = np.zeros_like(Memp_all)
        Menc_all = np.zeros(len(times))
        # Initilalize Basis
        coef_init = halo_basis.createFromArray(np.ones(1), [10, 10, 10], time=0.0)
        halo_coefs = pyEXP.coefs.Coefs.makecoefs(coef_init, halo)
        for t in range(Nsnaps):
            snap = int(init_halo_snap + t)
            print('-> Working with snapshot {}'.format(snap))
            data = np.load(DATAPATH + "{}/{}_{:03d}.npy".format(halo, halo, snap))
            h = {'position': data['x'], 'velocity': data['v']}
            rhalo = np.sqrt(np.sum(h['position']**2, axis=1))
            enc = np.where(rhalo<rvals_lin[-1])
            nparticles = len(h['position'])
            

            Menclosed = len(enc[0])*mp
            Mtot = nparticles*mp
            Mfrac = Menclosed/Mtot
       
                   

            Remp_all[t], Demp_all[t], Memp_all[t], Pemp_all[t] = makemodel.makemodel(makemodel.empirical_density_profile, M=1,
                                                                            funcargs=(h['position'], np.ones(len(rhalo))/nparticles, smooth), 
                                                                            rvals=rvals_lin)
            
            snap_coefficients = halo_basis.createFromArray(np.ones(nparticles), h['position'], time=t)
            halo_coefs.add(snap_coefficients)
        
        halo_coefs.WriteH5Coefs('MWest_coefficients_{}.h5'.format(halo))



        np.savetxt('denstiy_profiles_{}.txt'.format(halo), Demp_all)
        np.savetxt('potential_profiles_{}.txt'.format(halo), Pemp_all)
        np.savetxt('mass_profiles_{}.txt'.format(halo), Memp_all)
        np.savetxt('bins_profiles_{}.txt'.format(halo), Remp_all)
        
        
        #profile_plot([Rnfw, Remp_all], [Dnfw, Demp_all])

        #symphony_coef = symphony_basis.createFromArray(np.ones(1), [10, 10, 10], time=times[0])
        #halo_coefs = pyEXP.coefs.Coefs.makecoefs(symphony_coef, 'halo')
        
        #plt.loglog(Rnfw, Dnfw)
        #plt.loglog(Remp, Demp)
        #plt.loglog(Remp_smooth, Demp_smooth)
        