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
from scipy.optimize import  curve_fit

class OrdinaryKriging(object):
    def __init__(self):
        self.ARRAY_FILE_DELIM = ','
        self.MAP_VISUALIZER_TICK_LEN_METER = 100
        self.UNKNOWN_VALUE = -1000.
        self.OK_TX_APPROX_LEN_METER = 10
        self.OK_PATHLOSS_EXPONENT = 2.1

        self.config = ConfigParser()
        self.config.read('ordinary_kriging.ini')
        self._load_map_config()

        if 'y' in self.config['DEFAULT']['generate_spectrum_map']:
            self._generate_spectrum_map_data()
        if 'y' in self.config['DEFAULT']['generate_sparse_data']:
            self._sample_data_from_spectrum_map()
        self._load_spectrum_map_data()
        self._load_sparse_data()

        if 'y' in self.config['VISUALIZER']['visualize']:
            self._visualize_map( self.sparse_map, self.config['VISUALIZER']['output_sparse_data_filename'])

        if 'y' in self.config['VISUALIZER']['visualize']:
            self._visualize_map(self.spectrum_map, self.config['VISUALIZER']['output_spectrum_map_filename'])

        self._run_kriging()
        return

    def _load_map_config(self):
        self.number_of_transmitters =  int(self.config['SPECTRUM-MAP']['number_of_transmitters'])
        self.x_extent_meter = float(self.config['SPECTRUM-MAP']['x_extent_meter'])
        self.y_extent_meter = float(self.config['SPECTRUM-MAP']['y_extent_meter'])
        self.map_granularity_grid_per_meter = float(self.config['SPECTRUM-MAP']['granularity_sample_per_meter'])
        self.map_dim1 = int(round(self.x_extent_meter * self.map_granularity_grid_per_meter, 0))
        self.map_dim2 = int(round(self.y_extent_meter * self.map_granularity_grid_per_meter, 0))
        return

    def _load_spectrum_map_data(self):
        fname = str(self.config['SPECTRUM-MAP']['output_spectrum_map_filename'])
        self.spectrum_map = np.loadtxt(fname, dtype=np.float, delimiter=self.ARRAY_FILE_DELIM)
        self.map_dim1, self.map_dim2 = self.spectrum_map.shape
        return

    def _generate_pathloss_data(self):
        if 'y' in self.config['SPECTRUM-MAP']['generate_pathloss_from_log_normal']:
            self._generate_lognormal_pathloss_data()

        elif  'y' in self.config['SPECTRUM-MAP']['generate_pathloss_from_splat']:
            self._generate_splat_pathloss_data()
        return

    def _generate_splat_pathloss_data(self):
        #TODO: generate and save splat data for the ini file params
        return

    def _generate_lognormal_pathloss_data(self):
        self.pathloss_per_tx = []
        self.tx_x_locs = []
        self.tx_y_locs = []

        self.lognormal_pathloss_exponent = float(self.config['LOG-NORMAL-PATHLOSS']['pathloss_exponent'])
        for cur_tx_indx in range(self.number_of_transmitters):
            cur_pathloss_map = np.zeros((self.map_dim1, self.map_dim2), dtype=np.float)
            self.tx_x_locs.append(float(self.config['TX-' + str(cur_tx_indx + 1)]['x_loc_meter']) )
            self.tx_y_locs.append(float(self.config['TX-' + str(cur_tx_indx + 1)]['y_loc_meter']))

            tx_x_grid = int(round(self.tx_x_locs[cur_tx_indx] * self.map_granularity_grid_per_meter, 0))
            tx_y_grid = int(round(self.tx_y_locs[cur_tx_indx] * self.map_granularity_grid_per_meter, 0))

            for x_grid in range(self.map_dim1):
                for y_grid in range(self.map_dim2):
                    dx = 1.*np.abs(x_grid - tx_x_grid)/self.map_granularity_grid_per_meter
                    dy = 1.*np.abs(y_grid - tx_y_grid)/self.map_granularity_grid_per_meter
                    d = dx**2. +dy**2.
                    path_loss_dB = float('inf')
                    if d <= 0.:
                        path_loss_dB = 0
                    else:
                        path_loss_dB = -10.*(self.lognormal_pathloss_exponent/2.0)*np.log10(d)
                    cur_pathloss_map[x_grid, y_grid]  = path_loss_dB
            self.pathloss_per_tx.append(cur_pathloss_map)
        return

    def _generate_spectrum_map_data(self):
        self._generate_pathloss_data()
        self.spectrum_map = np.zeros(self.pathloss_per_tx[0].shape, dtype=np.float)
        self.tx_power_dB = []
        for cur_tx_indx in range(self.number_of_transmitters):
            self.tx_power_dB.append( float(self.config['TX-' + str(cur_tx_indx + 1)]['tx_power_dB'] ))
            self.spectrum_map += ( self.tx_power_dB[cur_tx_indx] + self.pathloss_per_tx[cur_tx_indx])
        self.noise_variance_dB = float(self.config['SPECTRUM-MAP']['noise_variance_dB'])
        self.spectrum_map -= np.random.rand(self.map_dim1, self.map_dim2)*self.noise_variance_dB
        self.lognormal_output_filename = self.config['SPECTRUM-MAP']['output_spectrum_map_filename']
        np.savetxt(self.lognormal_output_filename, self.spectrum_map, fmt="%.5f", delimiter=self.ARRAY_FILE_DELIM)

        return

    def _set_visualizer_ticks(self):
        xtick_postions = [i for i in np.arange(0, self.map_dim1 + 1, self.map_granularity_grid_per_meter * self.MAP_VISUALIZER_TICK_LEN_METER)]
        xtick_labels =  [str(int(i / self.map_granularity_grid_per_meter)) for i in xtick_postions]

        ytick_postions = [i for i in np.arange(0, self.map_dim2 + 1, self.map_granularity_grid_per_meter * self.MAP_VISUALIZER_TICK_LEN_METER)]
        ytick_labels =  [str(int(i / self.map_granularity_grid_per_meter)) for i in ytick_postions]

        plt.xticks(xtick_postions, xtick_labels)
        plt.yticks(ytick_postions, ytick_labels)
        plt.xlabel("Meter")
        plt.ylabel("Meter")
        plt.colorbar().ax.set_ylabel('RX (dBm)', rotation=270, labelpad=20)
        return

    def _visualize_map(self, cur_map, ofname = None):
        plt.clf()
        plt.imshow(cur_map.T, origin='lower',cmap='viridis')
        self._set_visualizer_ticks()
        if ofname is None:
            plt.show()
        else:
            plt.savefig(ofname )
        return

    def _sample_data_from_spectrum_map(self):
        all_grid_indices = [(i,j) for i in range(self.map_dim1) for j in range(self.map_dim2)]
        sample_size_percent = float(self.config['DATA-GENERATION']['sample_density_percent'])
        sample_size_count = int( round(1.*self.map_dim1*self.map_dim2*sample_size_percent/100.,0))
        selected_indices = random.sample(all_grid_indices, k=sample_size_count)
        self.sample_data=[]

        with open(self.config['DATA-GENERATION']['output_sparse_data_file'], "w") as f:
            for v in selected_indices:
                self.sample_data.append(( v[0],v[1], self.spectrum_map[v[0], v[1]]))
                ftext = str(v[0])+', '+str(v[1])+', '+str(self.spectrum_map[v[0], v[1]])+'\n'
                f.write(ftext)
        return

    def _load_sparse_data(self):
        self.sparse_map = np.full((self.map_dim1, self.map_dim2), self.UNKNOWN_VALUE, dtype=np.float)
        self.sample_data = []
        with open(self.config['ORDINARY-KRIGING']['input_sparse_data_file'], 'r') as f:
            for line in f:
                if not line:
                    continue
                v =  line.strip().split(',')
                if len(v) <3:
                    continue
                x , y, val = int(v[0]), int(v[1]), float(v[2])
                self.sample_data.append((x,y,v))
                self.sparse_map[x, y] = val
        return

    def _find_max_value_in_subarray(self, map_val, x, y, extent):
        max_val = -float('inf')
        max_x, max_y = -1, -1
        for i in range(max(0, x-extent) , min(x+extent, map_val.shape[0])):
            for j in range(max(0, y-extent) , min(y+extent, map_val.shape[1])):
                if map_val[i, j] != self.UNKNOWN_VALUE and max_val <  map_val[i, j]:
                    max_x, max_y, max_val = i, j, map_val[i, j]
        return max_x, max_y

    def _cluster_sparse_data(self):
        self.ok_partition_count = int(self.config['ORDINARY-KRIGING']['partition_count'])
        self.ok_tx_x_indx = []
        self.ok_tx_y_indx = []
        for cur_tx_indx in range(self.ok_partition_count):
            approx_x_indx = int(round(float(self.config['OK-PARTITION-'+str(cur_tx_indx+1)]['approx_tx_x_meter'])\
                            *self.map_granularity_grid_per_meter,0))
            approx_y_indx = int(round(float(self.config['OK-PARTITION-'+str(cur_tx_indx+1)]['approx_tx_y_meter'])\
                            *self.map_granularity_grid_per_meter,0))
            approx_x_indx = min(max(approx_x_indx, 0), self.map_dim1)
            approx_y_indx = min(max(approx_y_indx, 0), self.map_dim2)
            ix, iy = self._find_max_value_in_subarray(self.sparse_map, approx_x_indx, approx_y_indx, self.OK_TX_APPROX_LEN_METER)
            self.ok_tx_x_indx.append(ix)
            self.ok_tx_y_indx.append(iy)
        plt.clf()
        plt.xlim(0, self.map_dim1)
        plt.ylim(0, self.map_dim2)
        color_indx = ['r', 'g', 'b']
        self.ok_partitions = np.zeros((self.map_dim1, self.map_dim2), dtype=np.int)


        for i in range(self.sparse_map.shape[0]):
            for j in range(self.sparse_map.shape[1]):
                max_rx_val = -float('inf')
                max_rx_tx_indx = -1
                for cur_tx_indx in range(self.ok_partition_count):
                    cur_tx_x, cur_tx_y = self.ok_tx_x_indx[cur_tx_indx], self.ok_tx_y_indx[cur_tx_indx]
                    d_sq =  ( (i - cur_tx_x)*self.map_granularity_grid_per_meter)**2. + \
                            ( (j - cur_tx_y)*self.map_granularity_grid_per_meter)**2.
                    rx_val = -float('inf')
                    if d_sq <=0:
                        rx_val = self.sparse_map[cur_tx_x, cur_tx_y]
                    else:
                        rx_val = self.sparse_map[ cur_tx_x ,  cur_tx_y ] - 10.*self.OK_PATHLOSS_EXPONENT/2.0*np.log10(d_sq)
                    if rx_val > max_rx_val:
                        max_rx_val, max_rx_tx_indx = rx_val, cur_tx_indx
                self.ok_partitions[i, j] = max_rx_tx_indx

        plt.imshow(self.ok_partitions.T, origin='lower', cmap='viridis')
        for cur_tx_indx in range(self.ok_partition_count):
            cur_tx_x, cur_tx_y = self.ok_tx_x_indx[cur_tx_indx], self.ok_tx_y_indx[cur_tx_indx]
            plt.scatter(cur_tx_x, cur_tx_y, s=20, c='white')
        plt.savefig('cluster.png')
        return

    def _func_exponential_variogram(self, x, c, a):
        if a==0:
            a=0.000000001
        return 1.*c*(1-np.exp(-1.*x/a))

    def _generate_variogram_func(self):
        self.variogram_func_params = []
        for partition_indx in range(self.ok_partition_count):
            cur_tuples = []
            h_vals = {}
            for cur_sample in self.sample_data:
                i, j = cur_sample[0], cur_sample[1]
                if self.ok_partitions[i, j]==partition_indx:
                    z = self.sparse_map[i, j]
                    if 'y' in self.config['ORDINARY-KRIGING']['is_detrended']:
                        d_sq = ((i-self.ok_tx_x_indx[partition_indx])*self.map_granularity_grid_per_meter)**2. + \
                               ((j - self.ok_tx_y_indx[partition_indx]) * self.map_granularity_grid_per_meter) ** 2.
                        if d_sq == 0.:
                            z = 0.
                        else:
                            z -= 10.*self.OK_PATHLOSS_EXPONENT/2.*np.log10(d_sq)
                    cur_tuples.append((i, j, z))
            for i, v1 in enumerate(cur_tuples):
                for j, v2 in enumerate(cur_tuples):
                    if i<j:
                        h = round(np.sqrt(((v1[0] - v2[0])*self.map_granularity_grid_per_meter)**2. + \
                            ((v1[1] - v2[1]) * self.map_granularity_grid_per_meter)** 2.),2)
                        h_val = (v1[2] - v2[2])**2.
                        if not h in h_vals.keys():
                            h_vals[h] = h_val
                        else:
                            h_vals[h] = (h_vals[h]+h_val)/2.
            xdata = []
            ydata = []
            for i in sorted(h_vals.keys()):
                xdata.append(i)
                ydata.append(h_vals[i])
            params, _ = curve_fit(self._func_exponential_variogram, xdata, ydata)
            self.variogram_func_params.append((params[0], params[1]))
        return

    def _find_nearest_neighbors(self, ix, iy, k):
        xs = []
        ys = []
        zs = []
        ds = []
        for v in self.sample_data:
            if not self.ok_partitions[v[0], v[1]] == self.ok_partitions[ix, iy]:
                continue
            xs.append(v[0])
            ys.append(v[1])
            zs.append(v[2])
            d = (ix-v[0])**2.+(iy-v[1])**2.
            ds.append(d)
        ds_indices = np.array(ds).argsort()[-k:][::-1]
        nearest_xs, nearest_ys = [], []
        for ds_indx in ds_indices:
            nearest_xs.append(xs[ds_indx])
            nearest_ys.append(ys[ds_indx])

        return nearest_xs, nearest_ys

    def _interpolate_value(self, ix, iy, k):
        xs, ys = self._find_nearest_neighbors(ix, iy, k)
        n = len(xs)
        A = np.ones((n+1, n+1), dtype=np.float)
        B = np.ones((n+1, 1), dtype=np.float)
        for i in range(n):
            for j in range(n):
                if i == j:
                    h, h_val = 0., 0.
                else:
                    h = np.sqrt( ((xs[i] - xs[j])*self.map_granularity_grid_per_meter)**2. + \
                                       ((ys[i] - ys[j]) * self.map_granularity_grid_per_meter) ** 2. )
                    h_val = self._func_exponential_variogram(h, *self.variogram_func_params[self.ok_partitions[ix, iy]])
                A[i, j] = h_val
            h = np.sqrt( ((xs[i] - ix)*self.map_granularity_grid_per_meter)**2. + \
                                   ((ys[i] - iy) * self.map_granularity_grid_per_meter) ** 2. )
            h_val = 0.
            if h > 0.:
                h_val = self._func_exponential_variogram(h, *self.variogram_func_params[self.ok_partitions[ix, iy]])
            B[i] = h_val
        A[n, n] = 0.
        C = np.matmul( np.linalg.inv(A), B )
        interp_v = 0
        for i in range(n):
            interp_v += C[i]*self.sparse_map[xs[i], ys[i]]
        if 'y' in self.config['ORDINARY-KRIGING']['is_detrended']:
            partition_indx = self.ok_partitions[ix, iy]
            d_sq = ((ix - self.ok_tx_x_indx[partition_indx]) * self.map_granularity_grid_per_meter) ** 2. + \
                   ((iy - self.ok_tx_y_indx[partition_indx]) * self.map_granularity_grid_per_meter) ** 2.
            offset_val = 0.
            if d_sq > 0.:
                offset_val = 10. * self.OK_PATHLOSS_EXPONENT / 2. * np.log10(d_sq)
            interp_v += offset_val
        return interp_v

    def _run_kriging(self):
        if len(self.sample_data) <3:
            print("Not Enough data for kriging!!")
            return
        self._cluster_sparse_data()
        self._generate_variogram_func()

        self.nearest_neighbor_count = int(self.config['ORDINARY-KRIGING']['neighborhood_count'])
        for i in range(self.map_dim1):
            for j in range(self.map_dim2):
                if self.sparse_map[i, j] == self.UNKNOWN_VALUE:
                    self.sparse_map[i, j] = self._interpolate_value(i , j, self.nearest_neighbor_count)
                    print("debug: ",i,j,self.sparse_map[i, j])


        output_fname =  self.config['ORDINARY-KRIGING']['output_interpolated_data_file']
        np.savetxt(output_fname, self.sparse_map ,fmt="%.5f", delimiter=self.ARRAY_FILE_DELIM)
        if 'y'  in self.config['VISUALIZER']['visualize']:
            output_fname = self.config['VISUALIZER']['output_interpolated_data_filename']
            self._visualize_map(self.sparse_map, output_fname)
        return

if __name__ == '__main__':
    ok = OrdinaryKriging()