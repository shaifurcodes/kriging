#!/usr/bin/env python3
'''
mdsrahman@cs.stonybrook.edu,
August 2, 2020
rev: 2.0
'''
from configparser import  ConfigParser
import numpy as np
import matplotlib.pyplot as plt
import random

class OrdinaryKriging(object):
    def __init__(self):
        self.ARRAY_FILE_DELIM = ','

        self.config = ConfigParser()
        self.config.read('ordinary_kriging.ini')

        self._generate_pathloss_data()
        self._load_data()
        return

    def _load_data(self):
        fname = str(self.config['ORDINARY-KRIGING']['input_spectrum_map_file'])
        self.spectrum_map = np.loadtxt(fname, dtype=np.float, delimiter=self.ARRAY_FILE_DELIM)
        self.map_dim1, self.map_dim2 = self.spectrum_map.shape
        return

    def _generate_pathloss_data(self):
        self.number_of_transmitters =  int(self.config['PATHLOSS']['number_of_transmitters'])
        self.x_extent_meter = float(self.config['PATHLOSS']['x_extent_meter'])
        self.y_extent_meter = float(self.config['PATHLOSS']['y_extent_meter'])

        if 'y' in self.config['DEFAULT']['generate_pathloss_from_log_normal']:
            self._generate_lognormal_pathloss_data()
            self._generate_spectrum_map()
            if 'y' in self.config['VISUALIZER']['visualize']:
                self._visualize_map()
        elif  'y' in self.config['DEFAULT']['generate_pathloss_from_splat']:
            self._generate_splat_pathloss_data()
        return

    def _generate_splat_pathloss_data(self):
        #TODO: generate and save splat data for the ini file params
        return

    def _generate_lognormal_pathloss_data(self):
        self.pathloss_per_tx = []
        self.pathloss_granulairty = float(self.config['PATHLOSS']['granularity_sample_per_meter'])
        self.map_dim1 = int(round(self.x_extent_meter * self.pathloss_granulairty, 0))
        self.map_dim2 = int(round(self.y_extent_meter * self.pathloss_granulairty,0))
        self.tx_x_locs = []
        self.tx_y_locs = []

        self.lognormal_pathloss_exponent = float(self.config['LOG-NORMAL-PATHLOSS']['pathloss_exponent'])
        for cur_tx_indx in range(self.number_of_transmitters):
            cur_pathloss_map = np.zeros((self.map_dim1, self.map_dim2), dtype=np.float)
            self.tx_x_locs.append(float(self.config['TX-' + str(cur_tx_indx + 1)]['x_loc_meter']) )
            self.tx_y_locs.append(float(self.config['TX-' + str(cur_tx_indx + 1)]['y_loc_meter']))

            tx_x_grid = int(round(self.tx_x_locs[cur_tx_indx]*self.pathloss_granulairty, 0))
            tx_y_grid = int(round(self.tx_y_locs[cur_tx_indx]*self.pathloss_granulairty, 0))

            for x_grid in range(self.map_dim1):
                for y_grid in range(self.map_dim2):
                    dx = 1.*np.abs(x_grid - tx_x_grid)/self.pathloss_granulairty
                    dy = 1.*np.abs(y_grid - tx_y_grid)/self.pathloss_granulairty
                    d = dx**2. +dy**2.
                    path_loss_dB = float('inf')
                    if d <= 0.:
                        path_loss_dB = 0
                    else:
                        path_loss_dB = -10.*(self.lognormal_pathloss_exponent/2.0)*np.log10(d)
                    cur_pathloss_map[x_grid, y_grid]  = path_loss_dB
            self.pathloss_per_tx.append(cur_pathloss_map)
        return

    def _generate_spectrum_map(self):
        self.spectrum_map = np.zeros(self.pathloss_per_tx[0].shape, dtype=np.float)
        self.tx_power_dB = []
        for cur_tx_indx in range(self.number_of_transmitters):
            self.tx_power_dB.append( float(self.config['TX-' + str(cur_tx_indx + 1)]['tx_power_dB'] ))
            self.spectrum_map += ( self.tx_power_dB[cur_tx_indx] + self.pathloss_per_tx[cur_tx_indx])
        self.noise_variance_dB = float(self.config['PATHLOSS']['noise_variance_dB'])
        self.spectrum_map -= np.random.rand(self.map_dim1, self.map_dim2)*self.noise_variance_dB
        self.lognormal_output_filename = self.config['LOG-NORMAL-PATHLOSS']['output_filename']
        np.savetxt(self.lognormal_output_filename, self.spectrum_map, fmt="%.5f", delimiter=self.ARRAY_FILE_DELIM)
        return

    def _visualize_map(self):
        skip_len_meter = 100
        plt.imshow(self.spectrum_map, origin='lower',cmap='viridis', interpolation='nearest')
        xtick_postions = [i for i in np.arange(0, self.map_dim1+1, self.pathloss_granulairty*skip_len_meter)]
        xtick_labels =  [str(int(i/self.pathloss_granulairty)) for i in xtick_postions]

        ytick_postions = [i for i in np.arange(0, self.map_dim2+1, self.pathloss_granulairty*skip_len_meter)]
        ytick_labels =  [str(int(i/self.pathloss_granulairty)) for i in ytick_postions]

        plt.xticks(xtick_postions, xtick_labels)
        plt.yticks(ytick_postions, ytick_labels)
        plt.xlabel("Meter")
        plt.ylabel("Meter")
        plt.colorbar().ax.set_ylabel('RX (dBm)', rotation=270, labelpad=20)
        fname = str( self.config['VISUALIZER']['output_filename'] )
        plt.savefig(fname)
        plt.close()
        return

    def _sample_dataf_from_spectrum_map(self):
        all_grid_indices = [(i,j) for i in range(self.map_dim1) for j in range(self.map_dim2)]
        sample_size_percent = float(self.config['ORDINARY-KRIGING']['sample_density_percent'])
        sample_size_count = int( round(1.*self.map_dim1*self.map_dim2*sample_size_percent/100.,0))
        selected_indices = random.sample(all_grid_indices, k=sample_size_count)
        self.sample_data=[]
        for v in selected_indices:
            self.sample_data.append(( v[0],v[1], self.spectrum_map[v[0], v[1]]))
        return

    def run_kriging(self):
        self._sample_dataf_from_spectrum_map()
        if len(self.sample_data) <3:
            print("Not Enough data for kriging!!")
            return
        #TODO: code kriging algo
        return
if __name__ == '__main__':
    ok = OrdinaryKriging()
    ok.run_kriging()