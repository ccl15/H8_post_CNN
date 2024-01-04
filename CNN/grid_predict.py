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
            dcsr = csr - H8
            images.append( np.stack([H8, dcsr], axis=-1) )
            lat_norm = 1-i/525.0
            lon_norm = j/575.0
            attrs.append( np.array([np.sin(jday), np.cos(jday), hr, lon_norm, lat_norm]).astype(np.float32) )
    return np.array(images), np.array(attrs)


if __name__ == '__main__':
    exp = 'C12_dcsr' # saved name
    # model 
    model_settings = {'name':'models.CNN_1_2', 
                    'weight':'saved_models/CSR/C12_dcsr_d8_p6', 
                    'parameters':{
                        'filters': 64,
                        'levels': 4,
                        'FC_units': [128, 32,8],
                    }}
    model = create_model(model_settings)

    # do predict
    pred_dict = {}
    dates = [f'2020{yr:02d}20' for yr in range(3,13,3)]
    for date in dates:
        for hr in ['12','17']:
            time_str = date + hr
            images, attrs = read_data(time_str)
            predicts = []
            for i in range(509):
                pred = model(images[559*i:559*(i+1)], attrs[559*i:559*(i+1)])
                predicts.append(pred)
            pred_dict[time_str] = np.array(np.squeeze(predicts))
    np.save(f'../output/grid_data/{exp}.npy', pred_dict)
