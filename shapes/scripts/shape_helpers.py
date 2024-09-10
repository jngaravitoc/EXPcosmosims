import numpy as np
import matplotlib.pyplot as plt

def profiles_multipanel(shape, profiles, figname):
    """
    profiles multipanel
    """
    rbins = shape[0]
    ca = shape[1]
    ba = shape[2]
    theta = shape[3]
    fig, ax = plt.subplots(2, 2, figsize=(10, 5), sharex=True)
    ax[0][0].plot(rbins[:-1], np.log10(profiles['density']))
    ax[0][1].plot(rbins[:-1], profiles['beta'])
    
    #ax[1][0].plot(profiles['Q'])

    ax[1][0].plot(rbins, ca)
    ax[1][0].plot(rbins, ba)
    ax[1][1].plot(rbins, theta*180/np.pi)
    plt.savefig(figname, bbox_inches='tight')
    plt.close()

def save_properties(shape, profiles, filename):
    np.save("shape_"+filename, shape[:4])
    np.save("profiles_"+filename, np.array([profiles['density'], profiles['beta']]).T)
