import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import importlib
import numpy as np
from pathlib import Path
from datetime import datetime
#import tensorflow as tf


def create_model(model_settings):
    model = importlib.import_module(model_settings['name']).Model(**model_settings['parameters'])
    model.load_weights(model_settings['weight']+'/M').expect_partial()
    return model

def read_data(time_str):
    H8_file = f'/NAS-Kumay/H8/{time_str[:4]}/insotwf1h_{time_str}' 
    if not Path(H8_file).exists():
        print(f'H8 data does not exist: {H8_file}')
        return None
    csr_file = f'/NAS-Kumay/H8/clearsky_1HR/2019{time_str[4:6]}/insocld1h_2019{time_str[4:]}'
    if not Path(csr_file).exists():
        print(f'CSR data does not exist: {csr_file}')
        return None
    H8_data =  np.fromfile(H8_file, dtype=np.float32).reshape(525, 575).astype(np.float32)
    csr_data =  np.fromfile(csr_file, dtype=np.float32).reshape(525, 575).astype(np.float32)
    
    hr = int(time_str[8:])/24.0
    jday = int(datetime.strftime(datetime.strptime(time_str, '%Y%m%d%H'), '%j'))/365*2*np.pi
    dn = 4
    images = []
    attrs = []
    for i in range(8,517):
        for j in range(8,567):
            H8 = H8_data[i-dn:i+dn, j-dn:j+dn]
            csr = csr_data[i-dn:i+dn, j-dn:j+dn]
            images.append( np.stack([csr, H8], axis=-1) )
            #images.append( H8 )
            
            lat_norm = 1-i/525.0
            lon_norm = j/575.0
            attrs.append( np.array([np.sin(jday), np.cos(jday), hr, lon_norm, lat_norm]).astype(np.float32) )
    return np.array(images), np.array(attrs)


if __name__ == '__main__':
    model_settings = {'name':'models.CNN_1_2', 
                    'weight':'saved_models/CSR/C12_64_4_3_p2', 
                    'parameters':{
                        'filters': 64,
                        'levels': 4,
                        'FC_units': [128, 32,8],
                    }}
    model = create_model(model_settings)
    for date in ['20200620', '20201220']:
        for hr in ['12','17']:
            time_str = date + hr
            print(time_str)
            images, attrs = read_data(time_str)
            predicts = []
            for i in range(509):
                pred = model(images[559*i:559*(i+1)], attrs[559*i:559*(i+1)])
                predicts.append(pred)

            np.save(f'../output/grid_ssi/{time_str}.npy', predicts)
