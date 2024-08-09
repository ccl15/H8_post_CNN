import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
                        

#%% read ssi data as datafram 
def read_station():
    files = list(Path('1obs_h8_QC/').glob('*.npy'))
    dfs = pd.DataFrame()
    for file in files:
        data = np.load(file)
        sids = file.stem
        time = data[:,0]
        ssi = data[:,1].astype(float)
        df = pd.DataFrame({'sid':sids, 'time':time, 'ssi':ssi})
        dfs = pd.concat([dfs,df], ignore_index=True)
    dfs['time'] = pd.to_datetime(dfs['time'], format='%Y%m%d%H')
    #df.sort_values(by=['time'], inplace=True)
    return dfs

#%% Step 3. Combine with H8
def _station_lonlathigh():
    d2 = np.loadtxt('Stn_46_agr.txt', dtype=str)
    station_llh = {}
    for i in range(1, d2.shape[0]):
        station_llh[d2[i,0]] = [float(d2[i,1]), float(d2[i,2]), float(d2[i,3])]
    return station_llh

def _obs_loc_index(lon, lat):
    ilon = round((lon-117.78)/0.01)
    ilat = round((26.72-lat)/0.01)
    return ilon, ilat

def combine_H8_SSi_to_TFRecord(dfs):
    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    dn = 4
    for yr in range(2022,2024):
        # get the set of df time start with yr 
        times = dfs[dfs['time'].dt.year == yr]['time'].unique()
        tfname = f'2TFR_v3/H8csr_s8_{yr}.tfr'
        daysInYear = 366 if yr%4==0 else 365
        n=0
        
        with tf.io.TFRecordWriter(tfname) as writer:    
            for t1 in times:
                tstr1 = t1.strftime('%Y%m%d%H')
                tstr2 = t1.strftime('%m%d_%H')
                jday = int(datetime.strftime(t1, '%j'))/daysInYear*2*np.pi
                hr = t1.hour/24.0

                #H8_file = f'{H8_dir}/{tstr[:6]}/insotwf1h_{tstr}.dat'
                H8_file = f'{H8_dir}/{yr}/insotwf1h_{tstr1}'
                csr_file = f'{csr_path}/{t1.month:02d}/ClearSky_inso1hr_{tstr2}'
                if not (Path(H8_file).exists() and Path(csr_file).exists()):
                    print(f'{tstr1} not exist')
                    continue
                
                H8_data = np.fromfile(H8_file, dtype=np.float32).reshape(525, 575)
                csr_data = np.fromfile(csr_file, dtype=np.float32).reshape(525, 575)
            
                # get data 
                cases_intime = dfs[dfs.time == t1]
                for _, row in cases_intime.iterrows():
                    
                    slon, slat, _ = sid_llh[row.sid]
                    ilon, ilat = _obs_loc_index( slon, slat )
                    
                    H8_sta = H8_data[ilat-dn:ilat+dn, ilon-dn:ilon+dn].astype(np.float32)
                    csr_sta = csr_data[ilat-dn:ilat+dn, ilon-dn:ilon+dn].astype(np.float32)
                    
                    attr = np.array([np.sin(jday), np.cos(jday), hr, (slon-117.78)/5.75, (slat-21.47)/5.25]).astype(np.float32)
                    # write data
                    feature = {
                        'H8' : _bytes_feature( H8_sta.tobytes() ),
                        'ssi' : _float_feature( row.ssi ),
                        'attr': _bytes_feature( attr.tobytes() ),
                        'csr': _bytes_feature( csr_sta.tobytes() ),
                        'time': _bytes_feature( tstr.encode('utf-8') )
                    }
                    example = tf.train.Example(features=tf.train.Features(feature=feature))
                    writer.write(example.SerializeToString())
                    n += 1
        print(yr, n)

#%%
if __name__ ==  '__main__':
    H8_dir = '/NAS-Kumay/H8'
    csr_path = '/NAS-Kumay/H8/clearsky_1HR'
    lon = np.arange(117.78, 123.53, 0.01)
    lat = np.arange(26.72, 21.47, -0.01)
    sid_llh = _station_lonlathigh()

    dfs = read_station()
    combine_H8_SSi_to_TFRecord( dfs)
    
