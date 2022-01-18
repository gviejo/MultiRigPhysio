# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-01-10 12:01:26
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-01-12 19:17:01

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


# for s in info.index[0:5]:
for s in ['A8608-220110']:
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

	for i in range(len(idx_eps)):

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

		neurons = msi[msi>0.3].index.values

		# for k in range(len(tcurves)):
		# 	figure()
		# 	count = 1			
		# 	for i, n in enumerate(tcurves[k].columns):
		# 		subplot(int(np.sqrt(len(spikes)))+1,int(np.sqrt(len(spikes)))+1,count, projection = 'polar')
		# 		plot(tcurves[k][n])
		# 		if n in neurons:
		# 		 	plot(tcurves[k][n], linewidth = 3)
		# 		xticks([])
		# 		yticks([])				
		# 		count += 1

		# show()		


		# sys.exit(0)

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
			newep = velocity.threshold(0.06).time_support
			rate = nap.TsdFrame(rate).restrict(newep)

			data.append(rate.values)
			sessions.append(np.ones(len(rate))*j)

		data = np.vstack(data)
		sessions = np.hstack(sessions)

		# ISOMAP 2d
		ump2d = Isomap(n_components = 2, n_neighbors = 5).fit_transform(data)
		ump3d = Isomap(n_components = 3, n_neighbors = 50).fit_transform(data)

		p2d[s+'_'+str(i)] = ump2d
		p3d[s+'_'+str(i)] = ump3d
		pix[s+'_'+str(i)] = sessions

		# fig = figure()
		ax = fig.add_subplot(1,2,1)
		ax.scatter(ump2d[:,0], ump2d[:,1], c = sessions)
		ax = fig.add_subplot(1,2,2, projection='3d')
		ax.scatter(ump3d[:,0], ump3d[:,1], ump3d[:,2], c = sessions)
		show()
		sys.exit()
