import numpy as np
import statsmodels.api as sm
from scipy.interpolate import interpn
import matplotlib.pyplot as plt
import matplotlib.colors as mpc
plt.rcParams.update({'font.size': 12})

s, m, l = 6, 8, 10
plt.rc('figure', titlesize= l) # title
plt.rc('axes', labelsize= l) # xy labels
plt.rc('font', size= m)
plt.rc('xtick', labelsize= s)
plt.rc('ytick', labelsize= s)

def RMSE(yp,yt):
    return (np.mean((yp-yt)**2))**0.5

def MAE(yp,yt):
    return np.mean(np.abs(yp-yt))

def Bias(yp,yt):
    return np.mean(yp-yt)


class Scatter():
    def load_label(self):
        self.label = np.load(f'label/label_v3_{yr}.npy')

    def load_h8(self):
        self.pred = np.load(f'label/input{yr}.npy')[:,4,4,1]

    def load_alltime(self, sub_exp):
        self.pred = np.load(f'{exp}/{sub_exp}{yr}.npy')

    def plot_scatter(self):
        label = self.label
        pred = self.pred

        # calculate
        rmse = RMSE(pred,label)
        mae = MAE(pred,label)
        bias = Bias(pred,label)
        res = sm.OLS(pred, sm.add_constant(label)).fit()
        x = np.arange(0,5)
        y = res.params[0] + res.params[1]*x
        # scatter
        data, xe, ye = np.histogram2d(label, pred, bins=50, density=True)
        z = interpn( ( 0.5*(xe[1:] + xe[:-1]) , 0.5*(ye[1:]+ye[:-1]) ), 
                    data , np.vstack([label, pred]).T , 
                    method = "splinef2d", bounds_error = False)
        idx = z.argsort()

        # plot
        vmin, vmax = 1e-5, 10
        plt.figure()
        plt.scatter(label[idx], pred[idx], c=z[idx], cmap='jet', norm=mpc.LogNorm(vmin=vmin, vmax=vmax), s=8)
        plt.colorbar(label='Data density')
        plt.plot([0,6],[0,6],'k-')
        plt.plot(x, y,'b-')
        plt.axis([0,5,0,5])
        plt.gca().set_aspect('equal', adjustable='box')
        # text
        plt.text(0.1, 4.8, f'Number = {len(label)}')
        plt.text(0.1, 4.6, f'RMSE = {rmse:.3f}')
        plt.text(0.1, 4.4, f'MAE = {mae:.3f}')
        plt.text(0.1, 4.2, f'Bias = {bias:.3f}')
        plt.text(0.1, 4.0, f'R2 = {res.rsquared:.3f}')
        plt.text(2, 1.5, f'y = {res.params[1]:.3f}x + {res.params[0]:.3f}')
        # 
        plt.xlabel('Station SSI (MJ.m^-2)')
        plt.ylabel('CNN SSI (MJ.m^-2)')
        plt.title(yr)
        plt.savefig(f'{exp}/ana_{sub_exp}_{yr}.png', dpi=200, bbox_inches='tight')
        plt.close()


#%%
if __name__ == '__main__':
    exp = 'CSR'
    scatterer = Scatter()
    
    for yr in [2022,2023]:
        scatterer.load_label()
        for sub_exp in ['V3_C13dcsr_p8']:
            scatterer.load_alltime(sub_exp)
            scatterer.plot_scatter()
