import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import matplotlib.colors as mpc
from scipy.interpolate import interpn


def _density(x, y):
    data, xe, ye = np.histogram2d(x, y, bins=50, density=True)
    z = interpn( ( 0.5*(xe[1:] + xe[:-1]) , 0.5*(ye[1:]+ye[:-1]) ),
                data , np.vstack([x, y]).T ,
                method = "splinef2d", bounds_error = False)
    idx = z.argsort()
    return x[idx], y[idx], z[idx]
    

def dcnn_dh8_VSobs():    
    dcnn = cnn - label
    dh8 = h8 - label
    x,y,z = _density(dh8, dcnn)
    plt.figure()
    plt.scatter(x,y,c=z, cmap='summer',norm=mpc.LogNorm(vmin=1e-5, vmax=10), s=8)
    plt.colorbar()
    plt.axhline(0, color='black',linewidth=0.5)
    plt.axvline(0, color='black',linewidth=0.5)
    plt.xlabel('H8-obs')
    plt.ylabel('CNN-obs')
    plt.title(yr)
    plt.axis([-4,4,-4,4])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'{exp}_{yr}_VSobs.png', dpi=200, bbox_inches='tight')


def cloud_effect():
    cloud = csr-h8
    diff = cnn-label
    
    fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(14, 9))
    axs = axs.ravel()
    plt.subplots_adjust(hspace=0.3,)
    for i in range(12):
        idx = (hr==i+7)
        x, y, z = _density(cloud[idx], diff[idx])
        sc = axs[i].scatter(x,y,c=z, cmap='summer',norm=mpc.LogNorm(vmin=1e-2, vmax=10), #vmin=1e-2, vmax=1,
                            s=10, edgecolor='none')

        axs[i].set_title(i+7)
        axs[i].set_xlim(0, 3.7)
        axs[i].set_ylim(-3.2, 3.2)
        axs[i].axhline(0, color='black',linewidth=0.5)
    
    # Create a common colorbar for all subplots
    cax = fig.add_axes([0.92, 0.125, 0.02, 0.75])
    cbar = fig.colorbar(sc, cax=cax, aspect=30, extend='both')

    fs = 15
    fig.text(0.52, 0.95, 'Hour', ha='center', fontsize=fs)
    fig.text(0.52, 0.05, 'CSR-H8', ha='center', fontsize=fs)
    fig.text(0.06, 0.5, 'CNN-station', va='center', rotation='vertical', fontsize=fs)
    plt.savefig(f'{exp}_{yr}_cloud.png', dpi=200, bbox_inches='tight')


def high_obs():
    plt.figure()
    idx = label.argsort()
    plt.scatter(cnn[idx], h8[idx], c=label[idx],
                s=10, edgecolor='none',
                cmap='hot_r', vmax=4.5, vmin=0 )
    plt.colorbar(label='Station SSI',extend='max')
    plt.plot([0,6],[0,6],'k-')
    plt.xlabel('CNN')
    plt.ylabel('H8')
    plt.axis([0,5.5,0,5.5])
    plt.gca().set_aspect('equal', adjustable='box')
    plt.savefig(f'{exp}_{yr}_high.png', dpi=200, bbox_inches='tight')

def high_obs_diff():
    x,y,z = _density(label, cnn-h8)
    plt.figure()
    plt.scatter(x,y,c=z, cmap='summer', norm=mpc.LogNorm(vmin=1e-5, vmax=10), 
                s=10, edgecolor='none')
    plt.axhline(0, color='black',linewidth=0.5)
    plt.xlabel('Station')
    plt.ylabel('CNN-H8')
    plt.axis([0,5,-3.5,3.5])
    plt.savefig(f'{exp}_{yr}_high_mod.png', dpi=200, bbox_inches='tight')

def CNN_obs():
    plt.figure()
    plt.scatter(cnn[idx], h8[idx], c=label[idx],
                s=10, edgecolor='none',
                cmap='hot_r', vmax=4.5, vmin=0 )
    x,y,z = _density(dh8, dcnn)
    plt.figure()
    plt.scatter(x,y,c=z, cmap='summer',norm=mpc.LogNorm(vmin=1e-5, vmax=10), s=8)

# loading
exp = 'CSR/C12_dcsr_d8_p'
yr = '2022'
cnn = np.load(f'{exp}{yr}.npy')
label = np.load(f'label/label_{yr}.npy')
images = np.load(f'label/input{yr}.npy')
csr = images[:,4,4,0]
h8 = images[:,4,4,1]
attrs = np.load(f'label/attr{yr}.npy')
hr = attrs[:,2]*24
del images, attrs

## plot
cloud_effect()
#high_obs_diff()   

