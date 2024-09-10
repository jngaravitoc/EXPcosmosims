import numpy as np
import pynbody
# local 
import pynbody_helpers

class Structure:
    def __init__(self, data, rin, rout, nbins):
        self.pos = np.array(data['pos'])
        self.vel = np.array(data['vel'])
        self.mass = np.array(data['mass'])
        self.npart = len(data['mass'])
        self.nbins = nbins
        self.rin = rin
        self.rout = rout
        self.rbins = np.linspace(rin, rout, nbins)
        
    def randomize(self, nsample):
        nrand = np.random.randint(0, nsample, nsample)
        return self.pos[nrand], self.vel[nrand], self.mass[nrand] 

    def randomize_pynbody(self):
        pos_rand, vel_rand, mass_rand = self.randomize(self.npart)
        ## phase-space Dictionary
        ps_dict = {'position': pos_rand, 'velocity': vel_rand, 'mass': mass_rand}
        data_pn = pynbody_helpers.halo_format(ps_dict)
        return data_pn
    
    def shape_rand(self, nshells, rin, rout, bins):
        data_rand = self.randomize_pynbody()
        shapes = pynbody.analysis.halo.halo_shape(data_rand, N=nshells, rin=rin, rout=rout, bins=bins)
        return shapes 
    
    def shape(self, nshells, rin, rout, bins):
        ps_dict = {'position': self.pos, 'velocity': self.vel, 'mass': self.mass}
        data_pn = pynbody_helpers.halo_format(ps_dict)
        shapes = pynbody.analysis.halo.halo_shape(data_pn, N=nshells, rin=rin, rout=rout, bins=bins)
        return shapes
    
    def profiles(self):
        pos_rand, vel_rand, mass_rand = self.randomize()
        ps = {'position': pos_rand, 'velocity': vel_rand, 'mass': mass_rand}
        profile = pynbody.analysis.halo.profiles.Profile(ps, ndim=3, bins=self.rbins)
        return profile['density'], profile['beta']
          
    def bootstrap_shapes(self, shape_type, bin_type, nsamples=5000):
        """
        nsamples
        """
        nshells = self.nbins
        ba = np.zeros((nsamples, nshells))
        ca = np.zeros((nsamples, nshells))
        theta = np.zeros((nsamples, nshells))
        shape_bins = np.zeros((nsamples, nshells))
        
        print("! Bootstrapping with N={}".format(nsamples))
        
        
        rin = self.rin
        rout = self.rout
            
        if shape_type == 'cumulative':
            rin = 0.1
        
        if nsamples==1:
            for n in range(nsamples):
                shape_bins[n], ba[n], ca[n], theta[n], _ = self.shape(nshells=nshells, rin=rin, rout=rout, bins=bin_type)
    
        else:
            for n in range(nsamples):
                shape_bins[n], ba[n], ca[n], theta[n], _ = self.shape_rand(nshells=nshells, rin=rin, rout=rout, bins=bin_type)
    
        
        ca_median = np.median(ca, axis=0)
        ca_std = np.std(ca, axis=0)
        ba_median = np.median(ba, axis=0)
        ba_std = np.std(ba, axis=0)
        theta_median = np.median(theta, axis=0)
        theta_std = np.std(theta, axis=0)
        
        return np.median(shape_bins, axis=0), [ca_median, ca_std], [ba_median, ba_std], [theta_median, theta_std]