# -*- coding: utf-8 -*-
# @Author: Guillaume Viejo
# @Date:   2022-01-11 15:50:18
# @Last Modified by:   Guillaume Viejo
# @Last Modified time: 2022-01-13 13:09:52

import numpy as np
import pandas as pd
import pynapple as nap
import sys, os

def computeAngularVelocity(angle, ep, bin_size = 300):
	"""this function only works for single epoch
	"""
	angle 			= angle.restrict(ep)
	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
	tmp2 			= tmp.rolling(window=100,win_type='gaussian',center=True,min_periods=1).mean(std=5.0)
	time_bins		= np.arange(tmp.index[0], tmp.index[-1]+bin_size, bin_size*1000) # assuming milliseconds
	index 			= np.digitize(tmp2.index.values, time_bins)
	tmp3 			= tmp2.groupby(index).mean()
	tmp3.index 		= time_bins[np.unique(index)-1]+bin_size/2
	tmp3 			= nap.Tsd(tmp3)
	tmp4			= np.diff(tmp3.values)/np.diff(tmp3.as_units('s').index.values)
	velocity 		= nap.Tsd(t=tmp3.index.values[1:], d = np.abs(tmp4))
	return velocity


def zscore_rate(rate):
	idx = rate.index
	cols = rate.columns
	rate = rate.values
	rate = rate - rate.mean(0)
	rate = rate / rate.std(0)
	rate = pd.DataFrame(index = idx, data = rate, columns = cols)
	return nap.TsdFrame(rate)


def computeSpatialInfo(tc, angle, ep):
	nb_bins = tc.shape[0]+1
	bins 	= np.linspace(0, 2*np.pi, nb_bins)
	angle 			= angle.restrict(ep)
	# Smoothing the angle here
	tmp 			= pd.Series(index = angle.index.values, data = np.unwrap(angle.values))
	tmp2 			= tmp.rolling(window=50,win_type='gaussian',center=True,min_periods=1).mean(std=10.0)
	angle			= nap.Tsd(tmp2%(2*np.pi))
	pf = tc.values
	occupancy, _ 	= np.histogram(angle, bins)
	occupancy = occupancy/occupancy.sum()
	occ = occupancy[:,np.newaxis]	
	f = np.sum(pf * occ, 0)
	pf = pf / f	
	logpf = np.log2(pf)
	logpf[np.isinf(logpf)] = 0.0
	SI = np.sum(occ * pf * logpf, 0)
	SI = pd.DataFrame(index = tc.columns, columns = ['SI'], data = SI)
	return SI

def splitWake(ep):
	if len(ep) != 1:
		print('Cant split wake in 2')
		sys.exit()
	tmp = np.zeros((2,2))
	tmp[0,0] = ep.values[0,0]
	tmp[1,1] = ep.values[0,1]
	tmp[0,1] = tmp[1,0] = ep.values[0,0] + np.diff(ep.values[0])/2
	return nap.IntervalSet(start = tmp[:,0], end = tmp[:,1])