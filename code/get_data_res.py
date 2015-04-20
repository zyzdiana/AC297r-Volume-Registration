import os
import numpy as np
from utils import clean
def get_data():
	path = '/Users/zyzdiana/Dropbox/vNav_Test_Data/Apr_17_test_data/'
	dict_10mm = {}
	dict_6_4mm = {}
	dict_8mm = {}
	for root, dirs, files in os.walk(path):
		if len(dirs)==0:
			if('10mm' in root):
			    dict_10mm[root] = clean(files)
			if('6_4mm' in root):
			    dict_6_4mm[root] = clean(files)
			if('8mm' in root):
			    dict_8mm[root] = clean(files)

	list_10mm = []
	for item in dict_10mm.iteritems():
		list_10mm.append(os.path.join(item[0],item[1][0]))
	list_10mm.sort()
	vols_10mm = get_volume(list_10mm, 26)
	list_6_4mm = []
	for item in dict_6_4mm.iteritems():
		list_6_4mm.append(os.path.join(item[0],item[1][0]))
	list_6_4mm.sort()
	vols_6_4mm = get_volume(list_6_4mm, 40)
	list_8mm = []
	for item in dict_8mm.iteritems():
		list_8mm.append(os.path.join(item[0],item[1][0]))
	list_8mm.sort()
	vols_8mm = get_volume(list_8mm, 32)
	return vols_6_4mm, vols_8mm, vols_10mm
	

def get_volume(files, voxel):
	vol1 = np.memmap(files[4], dtype=np.complex64, mode='c', shape=(12,voxel,voxel,voxel))
	vol1 = np.array(np.linalg.norm(vol1,axis=0))
	vol2 = np.memmap(files[0], dtype=np.complex64, mode='c', shape=(12,voxel,voxel,voxel))
	vol2 = np.array(np.linalg.norm(vol2,axis=0))
	vol3 = np.memmap(files[2], dtype=np.complex64, mode='c', shape=(12,voxel,voxel,voxel))
	vol3 = np.array(np.linalg.norm(vol3,axis=0))
	vol4 = np.memmap(files[3], dtype=np.complex64, mode='c', shape=(12,voxel,voxel,voxel))
	vol4 = np.array(np.linalg.norm(vol4,axis=0))
	vol5 = np.memmap(files[1], dtype=np.complex64, mode='c', shape=(12,voxel,voxel,voxel))
	vol5 = np.array(np.linalg.norm(vol5,axis=0))
	return [vol1, vol2, vol3, vol4, vol5]




