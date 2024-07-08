import os
import argparse
import importlib
import numpy as np
from pathlib import Path
from datetime import datetime


def create_model():
    model = importlib.import_module('model.CNN_1_2').Model()
    model.load_weights('model/M').expect_partial()
    return model


def read_data(time_str):
    # images
    H8_file = f'/NAS-Kumay/H8/{time_str[:4]}/insotwf1h_{time_str}' 
    if not Path(H8_file).exists():
        print(f'H8 data does not exist: {H8_file}')
        return None, None
    
    csr_file = f'/NAS-Kumay/H8/clearsky_1HR/2019{time_str[4:6]}/insocld1h_2019{time_str[4:]}'
    if not Path(csr_file).exists():
        print(f'CSR data does not exist: {csr_file}')
        return None, None
    
    H8_data =  np.fromfile(H8_file, dtype=np.float32).reshape(525, 575).astype(np.float32)
    csr_data =  np.fromfile(csr_file, dtype=np.float32).reshape(525, 575).astype(np.float32)
    
    # attr
    hr = int(time_str[8:])/24.0
    jday = int(datetime.strftime(datetime.strptime(time_str, '%Y%m%d%H'), '%j'))/365*2*np.pi
    dn = 4
    
    # prepare input variable
    images = np.zeros((525-dn*2, 575-dn*2, dn*2, dn*2, 2), dtype=np.float32)
    attrs = np.zeros((525-dn*2, 575-dn*2, 5), dtype=np.float32)
    attrs[:,:,0:3] = [np.sin(jday), np.cos(jday), hr]
     
    for i in range(dn,525-dn):
        for j in range(dn,575-dn):
            H8 = H8_data[i-dn:i+dn, j-dn:j+dn]
            csr = csr_data[i-dn:i+dn, j-dn:j+dn]
            dcsr = csr - H8
            images[i-dn, j-dn, ...] = np.stack([H8, dcsr], axis=-1)
            lat_norm = 1-i/525.0
            lon_norm = j/575.0
            attrs[i-dn,j-dn,3:] = [lon_norm, lat_norm]
    return images, attrs


def predict_daily(day):
    # predict
    day_ssi = []
    predict_flag = False
    for hr in range(6, 20):
        hr_ssi = np.zeros((525, 575))
        hr_ssi.fill(np.nan)
        images, attrs = read_data(f'{day}{hr:02d}')

        if images is not None:
            predict_flag = True
            for i in range(517):
                pred = np.squeeze(model(images[i], attrs[i]))
                hr_ssi[i+4, 4:-4] = pred
        day_ssi.append(hr_ssi)

    # output
    outdir = Path(f'CNN_SSI/{args.day[:6]}')
    outdir.mkdir(parents=True, exist_ok=True)
    if predict_flag:
        np.save(f'{outdir}/{day}.npy', day_ssi)
        print(f'Save {day} done.')
    

#%%

if __name__ == '__main__':
    # date settings
    parser = argparse.ArgumentParser()
    parser.add_argument('day', type=str, help='%Y%m%d')
    #parser.add_argument('--end', type=str, default=None)
    parser.add_argument('-g', type=str, default='1', help='GPU numbers used.')
    args = parser.parse_args()

    # load model
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g
    model = create_model()

    # predict
    predict_daily(args.day)
