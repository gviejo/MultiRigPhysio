# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-01-13 13:00:21
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-01-13 13:39:57
# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-01-10 12:01:26
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-01-13 13:00:08

import numpy as np
import pandas as pd
import pynapple as nap
from pylab import *
from functions import *
from matplotlib.gridspec import GridSpecFromSubplotSpec

import sys, os
from matplotlib.colors import hsv_to_rgb
# import hsluv
from sklearn.manifold import Isomap
from mpl_toolkits.mplot3d import Axes3D

sheet_id = '1agZ8bdO_xdPq_l7_EUHefKDejQx9zVMKFbzPAKJtIe0'
sheet_name = 'A8608'
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"

info = pd.read_csv(url, index_col = [0], header = [1])
info = info.iloc[:,0:9]
info = info.dropna(0, 'all')

data_directory = '/mnt/Data2/PSB/A8608'

quadrant_dist = pd.DataFrame(columns = ['same', 'cross'])
ratio_dist = {}

p3d = {}
p2d = {}
pix = {}
bin_size = 0.3

numbers = pd.DataFrame(columns = ['n'])

for s in ['A8608-220108']:

	print(s)
	path = os.path.join(data_directory, s)
	data = nap.load_session(path, 'neurosuite')

	spikes = data.spikes
	position = data.position
	sleep_ep = data.epochs['sleep']
	wake_ep = data.epochs['wake']

	# Spliting between first and second exploration
	tmp = info.loc[s][info.loc[s] != 'sleep'].reset_index(drop=True).dropna()
	if s == 'A8608-220106':
		idx_eps = np.array_split(tmp.index.values, 2)
	else:
		idx_eps = np.array_split(tmp[tmp!='square'].index.values, 2)

	# for i in range(len(idx_eps)):
	for i in [1]:

		# Frequency
		ep = wake_ep.loc[idx_eps[i]].reset_index(drop=True).merge_close_intervals(0)
		neurons = list(spikes.restrict(ep).getby_threshold('freq', 0.5).keys())


		# Tuning curves
		tcurves = []
		spatial_info = {}
		for j in idx_eps[i]:
			tc = nap.compute_1d_tuning_curves(spikes, position['ry'], position.time_support.loc[[j]], 120)
			si = computeSpatialInfo(tc, position['ry'], position.time_support.loc[[j]])		
			tcurves.append(tc)
			spatial_info[j] = si['SI']
		spatial_info = pd.DataFrame.from_dict(spatial_info)

		msi = spatial_info.mean(1)

		msi = msi[neurons]

		neurons = msi[msi>0.1].index.values

		numbers.loc[s+'_'+str(i),'n'] = len(neurons)

		if len(neurons):
			# Binning
			data = []
			sessions = []
			angles = []
			for j in idx_eps[i]:
				count = spikes[neurons].count(bin_size, wake_ep.loc[[j]])
				count = count.as_dataframe()
				rate = np.sqrt(count/bin_size)
				rate = rate.rolling(window=100,win_type='gaussian',center=True,min_periods=1, axis = 0).mean(std=2)

				# refining with angular velocity
				velocity = computeAngularVelocity(position['ry'], wake_ep.loc[[j]], 300)
				newep = velocity.threshold(0.2).time_support
				rate = nap.TsdFrame(rate).restrict(newep)

				ep = wake_ep.loc[[j]]
				angle = position['ry'].restrict(ep)				
				bins = np.arange(ep.as_units('ms').start.iloc[0], ep.as_units('ms').end.iloc[-1]+bin_size, bin_size*1000)
				wakangle = pd.Series(index = np.arange(len(bins)-1), dtype = np.float)
				tmp = angle.as_series().groupby(np.digitize(angle.as_units('ms').index.values, bins)-1).mean()
				wakangle.iloc[tmp.index] = tmp
				wakangle.index = pd.Index(bins[0:-1] + np.diff(bins)/2.)
				wakangle = nap.Tsd(t = wakangle.index.values, d = wakangle.values, time_units = 'ms')
				wakangle = wakangle.restrict(newep)

				H = wakangle.values/(2*np.pi)
				HSV = np.vstack((H, np.ones_like(H), np.ones_like(H))).T
				RGB = hsv_to_rgb(HSV)

				angles.append(RGB)
				data.append(rate.values)
				sessions.append(np.ones(len(rate))*j)

			data = np.vstack(data)
			sessions = np.hstack(sessions)
			angles = np.vstack(angles)

			# ISOMAP 2d
			# ump2d = Isomap(n_components = 2, n_neighbors = 5).fit_transform(data)
			ump3d = Isomap(n_components = 3, n_neighbors = 50).fit_transform(data)

			# p2d[s+'_'+str(i)] = ump2d
			p3d[s] = ump3d
			pix[s] = sessions			


ratio_dist = pd.DataFrame.from_dict(ratio_dist)

sess = numbers[numbers>50].dropna().index.values


cmap = matplotlib.cm.get_cmap('viridis')
fig = figure()
gs = GridSpec(1,3)
gs2 = GridSpecFromSubplotSpec(2,1,gs[0,0])
subplot(gs2[0,0])
plot(position['x'].restrict(wake_ep.loc[[3]]), position['z'].restrict(wake_ep.loc[[3]]), color = 'yellow')
gca().set_aspect('equal')
subplot(gs2[1,0])
plot(position['x'].restrict(wake_ep.loc[[5]]), position['z'].restrict(wake_ep.loc[[5]]), color = 'darkblue')
gca().set_aspect('equal')

ax = fig.add_subplot(gs[0,1],projection='3d')
ax.scatter(p3d[s][:,0], p3d[s][:,1], p3d[s][:,2], color = angles)
title(s)


ax = fig.add_subplot(gs[0,2],projection='3d')
ax.scatter(p3d[s][:,0], p3d[s][:,1], p3d[s][:,2], c = pix[s])
title(s)

show()