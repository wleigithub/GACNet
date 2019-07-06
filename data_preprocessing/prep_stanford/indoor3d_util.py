import numpy as np
import glob
import os
import sys
import pdb
import h5py
import time
from plyfile import PlyData, PlyElement

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'eig_computation_op'))
from eig_feature import eig_cal_map, scene2blocks_withinner

# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------
    
def scene2blocks(data, label, num_point, block_size=1.0, inner_size=0.5, stride=0.5, count_threshold=100):

    assert(stride<=block_size)
    data[:,0:2] -= np.amin(data, 0)[0:2]
    pts_xyz = data[:,0:3]
    sample_idx, is_valid, block_num = scene2blocks_withinner(pts_xyz, num_point, block_size, inner_size, count_threshold)

    #horizen direction
    pts_xyz[:,0] += stride
    sample_idx_h, is_valid_h, block_num_h = scene2blocks_withinner(pts_xyz, num_point, block_size, inner_size, count_threshold)

    #diag direction
    pts_xyz[:,1] += stride
    sample_idx_d, is_valid_d, block_num_d = scene2blocks_withinner(pts_xyz, num_point, block_size, inner_size, count_threshold)

    #vihical direction
    pts_xyz[:,0] -= stride
    sample_idx_v, is_valid_v, block_num_v = scene2blocks_withinner(pts_xyz, num_point, block_size, inner_size, count_threshold)

    sample_idx = np.concatenate([sample_idx, sample_idx_h, sample_idx_v, sample_idx_d], 0)
    is_valid = np.concatenate([is_valid, is_valid_h, is_valid_d, is_valid_v], 0)
    block_num  = block_num+block_num_h+block_num_d+block_num_v

    data_batch = data[sample_idx, ...]
    label_batch = label[sample_idx, ...]
     
    data_batch = np.reshape(data_batch, [block_num, num_point, -1])
    label_batch = np.reshape(label_batch, [block_num, num_point, -1])
    is_valid = np.reshape(is_valid, [block_num, num_point])

    return data_batch, np.squeeze(label_batch), is_valid
            

def scene2blocks_zeromean(data_label_filename, num_point, block_size=1.0, inner_size=0.5, stride=0.5, radius=0.15):
    """ room2block, with input filename and RGB preprocessing.
        for each block centralize XY, z channel is keeped
        RGB to [0,1]
    """

    if data_label_filename[-3:] == 'txt':
        data_label = np.loadtxt(data_label_filename)
    elif data_label_filename[-3:] == 'ply':
        data_label = read_xyzrgbL_ply(data_label_filename)
    else:
        print('Unknown file type! exiting.')
        exit()

    print('data load done')
    data = data_label[:,0:6]
    data[:,3:6] /= 255.0
    label = data_label[:,-1].astype(np.uint8)
    idx = np.array(range(data.shape[0]))
    lalel_id = np.stack((label, idx), axis=-1)
    
    time_start = time.time()
    data_batch, label_id_batch, mask_batch = scene2blocks(data, lalel_id, num_point, block_size, inner_size, stride)
    #pdb.set_trace()
    batch_num = data_batch.shape[0]
    #zero mean
    for b in range(batch_num):
        mean_xy = np.mean(data_batch[b, :, 0:2], 0)
        data_batch[b, :, 0] -= mean_xy[0]
        data_batch[b, :, 1] -= mean_xy[1]
    #pdb.set_trace()
    #eigenvalues calculaition
    print('eigenvalues calculaiting...')
    idx = np.reshape(label_id_batch[:,:,-1], [-1])
    unique_idx = np.unique(idx)   
    eigs_batch = eig_cal_map(data[:,0:3], unique_idx, idx, radius)
    eigs_batch = np.reshape(eigs_batch, [batch_num, num_point, -1])
    data_batch = np.concatenate([data_batch, eigs_batch], -1)
    
    label_batch = label_id_batch[:,:,0]
    #pdb.set_trace()
    print('time for scene2block: %ds' %(time.time()-time_start))
    return data_batch, label_batch, mask_batch


# Write numpy array data and label to h5_filename
def save_h5(h5_filename, data, label, spw, data_dtype='float32', label_dtype='uint8', spw_dtype='float32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'data', data=data,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'label', data=label,
            compression='gzip', compression_opts=1,
            dtype=label_dtype)
    h5_fout.create_dataset(
            'spw', data=spw,
            compression='gzip', compression_opts=4,
            dtype=spw_dtype)
    h5_fout.close()


def read_xyzrgbL_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, red, green, blue, label] for x,y,z, red, green, blue, label in pc])
    return pc_array

