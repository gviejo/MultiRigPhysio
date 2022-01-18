# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-01-10 12:01:26
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-01-13 12:01:58

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

for s in info.index[0:5]:
# for s in [info.index[0:4][-1]]:
	print(s)
	path = os.path.join(data_directory, s)
	data = nap.load_session(path, 'neurosuite')

	spikes = data.spikes
	position = data.position
	sleep_ep = data.epochs['sleep']
	wake_ep = data.epochs['wake']

	

	# Spliting between first and second exploration
	tmp = info.loc[s][info.loc[s] != 'sleep'].reset_index(drop=True).dropna()
	envs = np.unique(tmp)

	idx_eps = [tmp[tmp==e].index.values for e in envs]

	for i in range(len(idx_eps)):

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

		numbers.loc[s+'_'+envs[i],'n'] = len(neurons)

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
				newep = velocity.threshold(0.25).time_support
				rate = nap.TsdFrame(rate).restrict(newep)

				data.append(rate.values)
				sessions.append(np.ones(len(rate))*j)

			data = np.vstack(data)
			sessions = np.hstack(sessions)

			# ISOMAP 2d
			# ump2d = Isomap(n_components = 2, n_neighbors = 5).fit_transform(data)
			ump3d = Isomap(n_components = 3, n_neighbors = 100).fit_transform(data)

			# p2d[s+'_'+str(i)] = ump2d
			p3d[s+'_'+envs[i]] = ump3d
			pix[s+'_'+envs[i]] = sessions

			fig = figure()
			ax = fig.add_subplot(1,2,1)
			# ax.scatter(ump2d[:,0], ump2d[:,1], c = sessions)
			ax = fig.add_subplot(1,2,2, projection='3d')
			ax.scatter(ump3d[:,0], ump3d[:,1], ump3d[:,2], c = sessions)
			# show()
			# sys.exit()

			#################################################################
			# Sphere ratio
			dist = np.sqrt(np.sum(np.power(ump3d[:,:,np.newaxis] - ump3d[:,:,np.newaxis].T, 2), 1))
			dist = dist/dist.max()

			idxes = np.unique(sessions).astype(np.int32) + 1
			sessions = sessions.astype(np.int32)

			tmp = np.tile(sessions[:,np.newaxis], len(sessions)) + 1
			tmp2 = tmp * tmp.T
			tmp3 = tmp2[np.triu_indices_from(tmp2,1)]
			tmp4 = dist[np.triu_indices_from(tmp2,1)]

			same_dist = np.hstack([tmp4[tmp3==k] for k in idxes**2])			
			cros_dist = np.hstack([tmp4[tmp3==k] for k in np.triu(idxes[:,np.newaxis]*idxes, 1).flatten()])


			bins = np.linspace(0, 1, 51)

			xs = np.array([np.sum(same_dist<t) for t in bins[1:]])
			xc = np.array([np.sum(cros_dist<t) for t in bins[1:]])
			# xs = np.histogram(same_dist, bins)[0]
			# xc = np.histogram(cros_dist, bins)[0]

			ratio_dist[s+'_'+envs[i]] = pd.Series(index = bins[0:-1], data = xs/(xs+xc))


			# #################################################################
			# # Quadrant binning
			# bins = np.linspace(-np.pi, np.pi, 13)
			# alpha = np.arctan2(ump2d[:,1], ump2d[:,0])		
			# idx = np.digitize(alpha, bins)-1

			# cross_distance = np.zeros(len(bins)-1)
			# same_distance = np.zeros(len(bins)-1)
			# for k in np.unique(idx):
			# 	pts = ump3d[idx == k]
			# 	idk = sessions[idx==k]
			# 	idist = []
			# 	for n in np.unique(idk):
			# 		tmp = pts[idk==n][:,:,np.newaxis] - pts[idk==n][:,:,np.newaxis].T
			# 		tmp2 = np.sqrt(np.sum(np.power(tmp,2),1))
			# 		idist.append(np.mean(tmp2[np.triu_indices_from(tmp2,1)]))
			# 	idist = np.mean(idist)
			# 	same_distance[k] = idist

			# 	if len(np.unique(idk)) == 2:
			# 		tmp = pts[idk == np.unique(idk)[0]][:,:,np.newaxis] - pts[idk == np.unique(idk)[1]][:,:,np.newaxis].T
			# 		cdist = np.sqrt(np.sum(np.power(tmp,2),1))
			# 		cdist = np.mean(cdist)
			# 		cross_distance[k] = cdist
					
			# 		#ratio_distance[k] = cdist/(cdist+idist)

			# quadrant_dist.loc[s+'_'+str(i),'same'] = same_distance.mean()
			# quadrant_dist.loc[s+'_'+str(i),'cross'] = cross_distance.mean()
			# ##############################################################

ratio_dist = pd.DataFrame.from_dict(ratio_dist)

# figure()
# for i,s in enumerate(p2d):
# 	ax = subplot(3,3,i+1)
# 	ax.scatter(p2d[s][:,0], p2d[s][:,1], c = pix[s])
# 	title(s)

numbers.index = pd.Index(list(p3d.keys()))
sess = numbers[numbers>50].dropna().index.values


fig = figure()
for i,s in enumerate(sess):	
	ax = fig.add_subplot(3,3,i+1,projection='3d')
	ax.scatter(p3d[s][:,0], p3d[s][:,1], p3d[s][:,2], c = pix[s])
	title(s)


cmap = matplotlib.cm.get_cmap('viridis')
figure()
subplot(121)
[plot(ratio_dist[sess[k]], color = cmap(k)) for k in range(len(sess))]
subplot(122)
imshow(ratio_dist.T)
show()

# figure()
# plot(quadrant_dist['same'], label = 'within')
# plot(quadrant_dist['cross'], label = 'across')
# xticks(rotation=45)
# legend()
# show()