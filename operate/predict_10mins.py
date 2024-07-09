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

class AttrLoader():
    def __init__(self):
        dn = 4 
        i_range = np.arange(dn, 525-dn+1)
        j_range = np.arange(dn, 575-dn+1)
        i_grid, j_grid = np.meshgrid(i_range, j_range, indexing='ij')

        attrs = np.zeros((525-dn*2+1, 575-dn*2+1, 5), dtype=np.float32)
        attrs[..., 3] = j_grid / 575.0
        attrs[..., 4] = 1 - i_grid / 525.0
        self.attrs = attrs
    
    def update_day(self, date):
        jday = int(datetime.strftime(datetime.strptime(date, '%Y%m%d'), '%j'))/365*2*np.pi
        self.attrs[:,:,0:2] = [np.sin(jday), np.cos(jday)]
    
    def get_update_attr(self, hh, mm):
        hr = (hh*60 + mm)/24.0/60.0
        self.attrs[:,:,2] = hr
        return self.attrs

    
def sliding_window_view(arr, window_shape):
    arr_shape = arr.shape
    window_shape = (arr_shape[0] - window_shape[0] + 1,
                    arr_shape[1] - window_shape[1] + 1) + window_shape
    strides = arr.strides + arr.strides
    return np.lib.stride_tricks.as_strided(arr, window_shape, strides)
    
def load_image(date, hh, mm):
    dn = 4
    # images
    H8_file = f'/NAS-Kumay/TAIPOWER/TPC_DATA/OBS/H8/{date[:4]}/{date}/insotwf10min_{date}{hh:02d}{mm:02d}' 
    if not Path(H8_file).exists():
        print(f'H8 data does not exist: {H8_file}')
        return None
    
    csr_file = f'/NAS-Kumay/H8/clearsky_10MIN/2019{date[4:6]}/insocldwatt_2019{date[4:]}_{hh:02d}{mm:02d}'
    if not Path(csr_file).exists():
        print(f'CSR data does not exist: {csr_file}')
        return None
    
    H8_data = np.fromfile(H8_file, dtype=np.float32).reshape(525, 575).astype(np.float32)
    csr_data = np.fromfile(csr_file, dtype=np.float32).reshape(525, 575).astype(np.float32)

    H8_windows = sliding_window_view(H8_data, (dn * 2, dn * 2))
    csr_windows = sliding_window_view(csr_data, (dn * 2, dn * 2))
    dcsr_windows = csr_windows - H8_windows

    images = np.stack([H8_windows, dcsr_windows], axis=-1)
    return images * 6


def predict_1hour(date, hh):
    # predict
    SSI_hr = []
    predict_flag = False

    for mm in range(0, 60, 10):
        # load image and attr
        images = load_image(date, hh, mm)
        attrs = attr_loader.get_update_attr(hh, mm)

        ssi = np.full((525, 575), np.nan)

        if images is not None:
            predict_flag = True
            for i in range(518):
                pred = np.squeeze(model(images[i], attrs[i]))
                ssi[i+4, 4:-3] = pred
        SSI_hr.append(ssi / 6)

    # output
    outdir = Path(f'CNN_SSI_10MIN/{date}')
    outdir.mkdir(parents=True, exist_ok=True)
    if predict_flag:
        np.save(f'{outdir}/{date}_{hh:02d}.npy', SSI_hr)
        print(f'Save {date} {hh} done.')
    

#%%

if __name__ == '__main__':
    # date settings
    parser = argparse.ArgumentParser()
    parser.add_argument('day', type=str, help='%Y%m%d')
    parser.add_argument('hr', type=int, help='hour')
    parser.add_argument('-g', type=str, default='1', help='GPU numbers used.')
    args = parser.parse_args()

    # load model
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.g
    model = create_model()

    # attrs template
    attr_loader = AttrLoader()
    attr_loader.update_day(args.day)

    # predict
    predict_1hour(args.day, args.hr)