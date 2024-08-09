import numpy as np
import statsmodels.api as sm
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
from matplotlib import colors, cm


def _obs_loc_index(lon, lat):
    ilon = round((lon-117.78)/0.01)
    ilat = round((26.72-lat)/0.01)
    return ilon, ilat

def clean_and_paired(sid, ilon, ilat):
    sta_ds = []
    for yr in range(2016, 2021):
        with open(f'{sta_dir}/{yr}/{sid}.txt', 'r') as f:
            next(f)
            for L in f:
                p = L.strip().split()
                hr = int(p[0][11:13])
                ssi = float(p[1])
                if hr>=6 and hr<=19 and ssi>=0:
                    # convert 2015-11-01 to tstr
                    tstr = f'{p[0][:4]}{p[0][5:7]}{p[0][8:10]}_{hr:02d}.dat'
                    H8_file = f'{H8_dir}/{yr}{p[0][5:7]}/insotwf1h_{tstr}'
                    if not Path(H8_file).exists():
                        continue
                    H8_data = np.fromfile(H8_file, dtype=np.float32).reshape(525, 575)
                    H8_staSSI = H8_data[ilat, ilon]
                    sta_ds.append([tstr, ssi, H8_staSSI])
                else:
                    continue
    sta_ds = np.array(sta_ds)
    np.save(f'1obs_h8_QC1620/{sid}.npy', sta_ds)
    print(sid, len(sta_ds))
    return sta_ds

def plot_paired(tstr, obs_ssi,h8_ssi):
    yrs = np.array([int(t[:4]) for t in tstr])
        
    plt.figure(figsize=(5,5))
    plt.plot([0,5],[0,5],'k-')
    cmap = colors.ListedColormap(cm.get_cmap("Paired").colors[1:9])
    plt.scatter(obs_ssi,h8_ssi, alpha=0.8, s=8, c=yrs, vmin=2015.5,vmax=2023.5, cmap=cmap )
    plt.xlabel('Station SSI (MJ/m^2)')
    plt.ylabel('H8 SSI (MJ/m^2)')
    plt.title(f'{sid}')
    plt.axis([-0.1,4,-0.1,4])
    plt.xticks(range(0,5))
    plt.yticks(range(0,5))
    plt.gca().set_aspect('equal', adjustable='box')
    
    res = sm.OLS(h8_ssi, sm.add_constant(obs_ssi)).fit()
    plt.text(0,3.6, f'Number = {len(obs_ssi)}')
    plt.text(0,3.4, f'R2 = {res.rsquared:.3f}')
    
    plt.colorbar(shrink=0.8)
    #cbar.set_label('Year')
    plt.savefig(f'1obs_h8_QC/scatter/{sid}.png', dpi=150, bbox_inches='tight')
    plt.close()

def diff_HM(tstr, diff):
    ym = np.array([t[:6] for t in tstr]  )

    diffs = np.zeros((8,12))
    # obs-H8
    for yr in range(2016,2024):
        for mon in range(1,13):
            mask = (ym == f'{yr}{mon:02d}')
            if mask.sum() == 0:
                diffs[yr-2016,mon-1] = np.nan    
            diffs[yr-2016,mon-1] = np.mean(diff[mask])
    
    masked_array = np.ma.array(diffs, mask=np.isnan(diffs))
    cmap = matplotlib.cm.bwr
    cmap.set_bad(color='#BDBDBD')

    im = plt.imshow(masked_array, vmin=-0.2, vmax=0.2, cmap=cmap, aspect='auto')#, extent=[1,12,0]9,)
    plt.colorbar(im,shrink=0.8, pad = 0.02, label='(MJ/m^2)', ticks=np.arange(-0.2,0.21,0.1), extend='both')
    
    plt.title(f'{sid} station-H8 SSI')
    plt.xlabel('Month')
    plt.ylabel('Year')
    plt.xticks(range(12),range(1,13))
    plt.yticks(range(8),range(2016,2024))
    plt.savefig(f'1obs_h8_QC/diff_ym/{sid}.png', dpi=150, bbox_inches='tight')
    plt.close()


out_staion = ['460010','460020','466850','466921','467411','467570','467790',
              '72L140','A2Q950','CAH030','E2S390','G2AI50','V2C250','V2C260','V2K610']

out_time = {
    '82H320': [f'{yr}{mm:02d}' for mm in range(1,13) for yr in range(2015,2019)],
    '467080': ['201706','201707'],
    'G2P820': ['201702','201708'],
    '466900': [f'2019{mm:02d}' for mm in range(6,10)],
    '467650': ['201605', '201702','201703','201704','201705']
}

sta_dir = '/NAS-DS1515P/users1/T1/API/Data/AllStn/hour/GlobalSolarRadiation_Accumulation'
H8_dir = '/NAS-Kumay/H8/ver202201_MODIS/inso1h'

d2 = np.loadtxt('Stn_46_agr.txt', dtype=str)
for i in range(1, d2.shape[0]):
    sid = d2[i,0]
    if sid in out_staion:
        continue
    
    lon = float(d2[i,1])
    lat = float(d2[i,2])
    ilon, ilat = _obs_loc_index(lon, lat)
    sta_ds = clean_and_paired(sid, ilon, ilat)

    #%%
    sid = d2[i,0]
    '''
    sta_ds = np.load(f'1obs_h8/{sid}.npy')
    if sid in out_time:
        sta_out_time =  out_time[sid]
        sta_out_time.append('201511')
    else:
        sta_out_time = ['201511']
    tlist, obsssi, h8ssi = sta_ds[:,0], sta_ds[:,1].astype(float), sta_ds[:,2].astype(float)
    mask_t = np.isin([t[:6] for t in tlist], sta_out_time, invert=True)
    mask_0 = h8ssi > 0
    mask_e = ~( (h8ssi>1) & (obsssi<0.1) )
    mask = (mask_t & mask_0 & mask_e)
    
    #plot_paired(tlist[mask], obsssi[mask],h8ssi[mask])
    #diff_HM(tlist[mask], obsssi[mask]-h8ssi[mask])
    #print(sid, sum(mask) )
    #np.save(f'1obs_h8_QC_1620/{sid}.npy', sta_ds[mask,:])
    '''


