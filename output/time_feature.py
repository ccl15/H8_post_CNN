import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({'font.size': 10})
#from collections import Counter



# data loading
yr = 2022
#sub_exp = 'CSR/C12_dcsr_d8_p'
#pred = np.load(f'{sub_exp}{yr}.npy') # load predict 
pred = np.load(f'label/input{yr}.npy')[:,4,4,1] # H8-SSI
label = np.load(f'label/label_{yr}.npy') # load test data

# load infomations
ds = np.load('label/date_hr_2022.npy')
dates = ds[0,:]
times = ds[1,:]


#%%  
def get_error():
    # calculate MAE at each date time box
    set_dates = range(1,366)
    set_times = range(6,20)

    mae  = np.zeros((14,366))
    bias = np.zeros((14,366))
    for i, date in enumerate(set_dates):
        for j, time in enumerate(set_times):
            mask = (dates==date) & (times==time)
            mae[j,i] = np.mean(np.abs(pred[mask]-label[mask]))
            bias[j,i] = np.mean(pred[mask]-label[mask])
    return mae, bias

def plot_mae(fig, axs, mae, title):
    im = axs.imshow(mae, cmap='Oranges', aspect='auto', extent=[0, 364, 19, 6], vmin=0, vmax=0.5)
    fig.colorbar(im, pad = 0.02, label='(MJ/m^2)', ticks=np.arange(0,0.6,0.1), extend='max')
    axs.set_title(title, loc='left')

def plot_bias(fig, axs, bias, title):
    im = axs.imshow(bias, cmap='bwr', aspect='auto', extent=[0, 364, 19, 6], vmin=-0.4, vmax=0.4)
    fig.colorbar(im, pad = 0.02, label='(MJ/m^2)', ticks=np.arange(-0.4,0.41,0.2),extend='both')
    axs.set_title(title, loc='left')

def full_plot():
    # Create plot
    fig, axs = plt.subplots(2)

    plot_mae(fig, axs[0], mae, 'CNN-Station MAE')
    plot_bias(fig, axs[1], bias, 'CNN-Station')
    
    for ax in axs:
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        
        ax.invert_yaxis()
        ax.set_xlabel('Month')
        ax.set_ylabel('Hour')
        ax.set_yticks(range(6,19,3), range(6,19,3))
    
        plt.subplots_adjust(hspace=0.8)
        plt.savefig(f'{sub_exp}_CNNobs.png', dpi=200, bbox_inches='tight')

mae, bias = get_error()
full_plot()