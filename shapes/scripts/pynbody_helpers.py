import numpy as np
import pynbody

def halo_format(particles, **kwargs):
    ndark = len(particles['mass'])
    halo_pynb = pynbody.new(dark=int(ndark))
    halo_pynb.dark['pos'] = particles['position']
    halo_pynb.dark['vel'] = particles['velocity']
    halo_pynb.dark['mass'] = particles['mass']
    
    if 'treeind' in kwargs:
         halo_pynb.dark['treeind'] = particles['treeind']
            
    return halo_pynb
