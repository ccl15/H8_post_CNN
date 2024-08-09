import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
plt.rcParams.update({'font.size': 10})
from datetime import datetime, timedelta
#from tqdm import tqdm
#from collections import Counter


#%%  
def get_error(yr):
    days = 366 if yr%4==0 else 365

    # calculate MAE at each date time box
    t0 = datetime(yr, 1, 1, 0, 0, 0)
    set_hr = range(6,19)

    mae  = np.zeros((14,days))
    bias = np.zeros((14,days))
    for i in range(days):
        for j, hr in enumerate(set_hr):
            t = t0 + timedelta(days=i, hours=hr)
            mask = times==t
            mae[j,i] = np.mean(np.abs(pred[mask]-label[mask]))
            bias[j,i] = np.mean(pred[mask]-label[mask])
    return mae, bias


def full_plot():
    # Create plot
    fig, axs = plt.subplots(2)

    x0 = mdates.date2num(datetime(yr, 1, 1))
    x1 = mdates.date2num(datetime(yr, 12, 31))

    im = axs[0].imshow(mae, cmap='Oranges', aspect='auto', extent=[x0, x1, 19, 5], vmin=0, vmax=0.5)
    fig.colorbar(im, pad = 0.02, label='(MJ/m^2)', ticks=np.arange(0,0.6,0.1), extend='max')
    axs[0].set_title(f'CNN-Station MAE', loc='left')
    axs[0].set_title(yr, loc='right')

    im = axs[1].imshow(bias, cmap='bwr', aspect='auto', extent=[x0, x1, 19, 5], vmin=-0.4, vmax=0.4)
    fig.colorbar(im, pad = 0.02, label='(MJ/m^2)', ticks=np.arange(-0.4,0.41,0.2),extend='both')
    axs[1].set_title(f'CNN-Station Bias', loc='left')
    axs[1].set_title(yr, loc='right')
    
    for ax in axs:
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13)))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
        ax.set_xlabel('Month')
        
        ax.invert_yaxis()
        ax.set_yticks(range(6,19,3), range(6,19,3))
        ax.set_ylabel('Hour')
    
        plt.subplots_adjust(hspace=0.8)
        plt.savefig(f'{sub_exp}_{yr}_ERROR.png', dpi=200, bbox_inches='tight')

# data loading
sub_exp = 'CSR/C12_dcsr_d8_p6'

for yr in [2022]:
    pred = np.load(f'{sub_exp}{yr}.npy') # load predict 
    label = np.load(f'label/label_{yr}.npy') # load test data

    # load infomations
    times = np.load(f'label/date_{yr}.npy', allow_pickle=True)

    # calculate and plotting 
    mae, bias = get_error(yr)
    full_plot()