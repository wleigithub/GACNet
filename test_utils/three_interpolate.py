from ctypes import *  
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

funs_c = CDLL(os.path.join(BASE_DIR,"build/libthree_interpolate.so")) 
def three_interpolate(data, idx, features, class_num):
    """
    input:
        data: (N,xyzrgb...)
        features: (M, class_num)
        idx: (M)
    return:
        labels: (N, class_num)
    """
    point_num = data.shape[0]
    sample_num = features.shape[0]

    xyz = data[:, 0:3]
    rgb = data[:, 3:6]/255
    xyz = np.reshape(xyz, [-1])
    rgb = np.reshape(rgb, [-1])
    features = np.reshape(features, [-1])

    xyz_c = (c_float*(point_num*3))(*xyz)
    rgb_c = (c_float*(point_num*3))(*rgb)
    features_c = (c_float*(sample_num*class_num))(*features)
    sample_id_c = (c_int*(sample_num*3))(*idx)
    point_num_c = (c_int*1)(point_num)
    sample_num_c = (c_int*1)(sample_num)

    class_num_c = (c_int*1)(class_num)

    probs = (c_float*(point_num*class_num))()

    funs_c.three_interpolate(xyz_c, rgb_c, features_c, sample_id_c, probs, point_num_c, sample_num_c, class_num_c)

    return np.reshape(probs, [point_num,-1])


###########################################################################################################
###########################             TEST 
###########################################################################################################
def label2probs(label, num_classes=13): # convert the certain label to probalities for optimazition
    '''
    default: label.dtype = np.int32
    '''
    num_label = label.shape[0]
    probs = np.ones([num_label, num_classes], dtype=np.float32)*0.05 #parameter is adjustable
    for i in range(num_label):
        probs[i, label[i]] = 0.95 #parameter is adjustable
    return probs

import provider

import pdb
if __name__=='__main__':
    datafilename = '/media/wl/myhome/wl/wanglei/Code/transequnet/Data/stanford_indoor/stanford_indoor3d/Area_1_copyRoom_1.ply' 
    #org_data = np.loadtxt(datafilename)
    org_data = provider.read_xyzrgbL_ply(datafilename)
    CLASS_NUM = 13
    pts_num = org_data.shape[0]
    idx = np.random.choice(pts_num, pts_num//5)

    label = np.int32(org_data[idx, -1])

    sub_probs = label2probs(label, CLASS_NUM)
    #pdb.set_trace()

    probs = three_interpolate(org_data, idx, sub_probs, CLASS_NUM)
    #pdb.set_trace()
    label = np.argmax(probs, axis=-1)
    label = np.expand_dims(label, -1)
    pred = np.concatenate([org_data[:,0:3], label], -1)
    
    np.savetxt('1.txt', pred)

