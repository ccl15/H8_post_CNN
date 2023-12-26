import numpy as np
import pandas as pd
import  matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader

class SSI_Ploter():
    def __init__(self):
        lon = np.arange(117.78, 123.53, 0.01)
        lat = np.arange(26.72, 21.47, -0.01)
        self.xx, self.yy = np.meshgrid(lon, lat)
        self._map_setting()
        
    def _map_setting(self):
        self.lon_min = 118
        self.lon_max = 123
        self.lat_min = 21.6
        self.lat_max = 26.5

        self.proj = ccrs.PlateCarree()
        reader_tw0    = Reader('/home/ccl/Resources/shp_file/gadm36_TWN_0.shp')
        reader_tw1    = Reader('/home/ccl/Resources/shp_file/gadm36_TWN_2.shp')
        self.tw_coastline  = cfeature.ShapelyFeature(reader_tw0.geometries(),
                                                     self.proj, edgecolor='k', facecolor='none')
        self.tw_countyline  = cfeature.ShapelyFeature(reader_tw1.geometries(),
                                                      self.proj, edgecolor='k', facecolor='none')
    
    def _load_h8(self, tstr):
        H8_file = f'/NAS-Kumay/H8/{tstr[:4]}/insotwf1h_{tstr}'
        self.h8data = np.fromfile(H8_file, dtype=np.float32).reshape(525, 575)
    def _load_cnn(self, tstr):
        cnn_data = np.full((525, 575), np.nan)
        cnn_data[8:517, 8:567] = np.squeeze(np.load(f'grid_ssi/{tstr}.npy'))
        self.cnn_data = cnn_data

    def plot_ssi(self, data, fig_name):
        vmax = 4.5
        
        fig, ax = plt.subplots(figsize=(5,5.5), subplot_kw={'projection': self.proj})
        ax.set_extent([self.lon_min, self.lon_max, self.lat_min, self.lat_max], crs=self.proj)

        plt.contourf(self.xx, self.yy, data, 
                    cmap='hot_r',
                    levels=np.arange(0, vmax+.1, 0.25),
                    extend='max')
        plt.colorbar(ticks=np.arange(0, vmax+.1, 1), shrink=0.6)

        ax.add_feature(self.tw_coastline, linewidth=0.2)
        ax.add_feature(self.tw_countyline, linewidth=0.2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks(np.arange(118, 124, 1))
        plt.yticks(np.arange(22, 27, 1))
        plt.savefig(f'grid_fig/{fig_name}.png', dpi=200, bbox_inches='tight')

    def plot_diff(self, tstr):
        data = self.cnn_data - self.h8data
        
        fig, ax = plt.subplots(figsize=(5,5.5), subplot_kw={'projection': self.proj})
        ax.set_extent([self.lon_min, self.lon_max, self.lat_min, self.lat_max], crs=self.proj)
        plt.contourf(self.xx, self.yy, data,
                     cmap='bwr',
                     levels= np.arange(-3, 3.1, 0.2),
                     extend='both')
        plt.colorbar(ticks=np.arange(-3, 3.1, 1), shrink=0.6)

        ax.add_feature(self.tw_coastline, linewidth=0.2)
        ax.add_feature(self.tw_countyline, linewidth=0.2)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.xticks(np.arange(118, 124, 1))
        plt.yticks(np.arange(22, 27, 1))
        plt.savefig(f'grid_fig/{tstr}d.png', dpi=200, bbox_inches='tight')


ssi_ploter = SSI_Ploter()
for date in ['20200620', '20201220']:
    for hr in ['12','17']: 
        tstr = date + hr
        ssi_ploter._load_cnn(tstr)
        ssi_ploter._load_h8(tstr)
        ssi_ploter.plot_ssi(ssi_ploter.cnn_data, f'{tstr}c')
        #ssi_ploter.plot_ssi(ssi_ploter.h8data, f'{tstr}h')
        ssi_ploter.plot_diff(tstr)
