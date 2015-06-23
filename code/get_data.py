import numpy as np
import os
from utils import clean

# Get All Volumes into a Dictionary
def idx_to_key(idx, keys):
    if(idx == 0):
        return keys[4]
    if(idx == 1):
        return keys[0]
    if(idx == 2):
        return keys[2]
    if(idx == 3):
        return keys[3]
    if(idx == 4):
        return keys[1]
    
def get_volume_1(path, voxel):
    volume = np.memmap(path, dtype=np.complex64, mode='c', shape=(12,voxel,voxel,voxel))
    volume = np.array(np.linalg.norm(volume,axis=0))
    return volume
    
def get_data_dict():
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
    keys_10 = sorted(dict_10mm.keys())
    keys_8 = sorted(dict_8mm.keys())
    keys_6_4 = sorted(dict_6_4mm.keys())
    dict_3res = {}
    dict_3res['6_4mm'] = {}
    dict_3res['8mm'] = {}
    dict_3res['10mm'] = {}
    for i in xrange(5):
        dict_3res['10mm'][idx_to_key(i, keys_10).split('/')[-1][5:]] = []
        dict_3res['8mm'][idx_to_key(i, keys_8).split('/')[-1][5:]] = []
        dict_3res['6_4mm'][idx_to_key(i, keys_6_4).split('/')[-1][5:]] = []
        for j in xrange(5):
            # 10mm
            path = os.path.join(idx_to_key(i, keys_10),dict_10mm[idx_to_key(i, keys_10)][j])
            dict_3res['10mm'][idx_to_key(i, keys_10).split('/')[-1][5:]].append(get_volume_1(path, 26))
            # 8mm
            path = os.path.join(idx_to_key(i, keys_8),dict_8mm[idx_to_key(i, keys_8)][j])
            dict_3res['8mm'][idx_to_key(i, keys_8).split('/')[-1][5:]].append(get_volume_1(path, 32))
            # 6.4mm
            path = os.path.join(idx_to_key(i, keys_6_4),dict_6_4mm[idx_to_key(i, keys_6_4)][j])
            dict_3res['6_4mm'][idx_to_key(i, keys_6_4).split('/')[-1][5:]].append(get_volume_1(path, 40))
    return dict_3res
    
def get_data_all():
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
    keys_10 = sorted(dict_10mm.keys())
    keys_8 = sorted(dict_8mm.keys())
    keys_6_4 = sorted(dict_6_4mm.keys())
    all_10mm = {}
    all_8mm = {}
    all_6_4mm = {}
    for i in xrange(5):
        all_10mm[idx_to_key(i, keys_10)] = []
        all_8mm[idx_to_key(i, keys_8)] = []
        all_6_4mm[idx_to_key(i, keys_6_4)] = []
        for j in xrange(5):
            # 10mm
            path = os.path.join(idx_to_key(i, keys_10),dict_10mm[idx_to_key(i, keys_10)][j])
            all_10mm[idx_to_key(i, keys_10)].append(get_volume_1(path, 26))
            # 8mm
            path = os.path.join(idx_to_key(i, keys_8),dict_8mm[idx_to_key(i, keys_8)][j])
            all_8mm[idx_to_key(i, keys_8)].append(get_volume_1(path, 32))
            # 6.4mm
            path = os.path.join(idx_to_key(i, keys_6_4),dict_6_4mm[idx_to_key(i, keys_6_4)][j])
            all_6_4mm[idx_to_key(i, keys_6_4)].append(get_volume_1(path, 40))
    return all_10mm, all_8mm, all_6_4mm

# Get one volume for each position
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

