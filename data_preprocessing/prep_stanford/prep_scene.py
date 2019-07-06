import numpy as np
import h5py
import os
import sys
import pdb
import glob
from plyfile import PlyData, PlyElement

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(os.path.join(os.path.dirname(BASE_DIR), 'eig_computation_op'))
from eig_feature import eig_cal_map, FPS

# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------
def test_area2blocks_zeromean(data_filename, num_point, block_size=1.0, inner_size=0.5, radius=0.15):
    if data_filename[-3:] == 'ply':
        data = read_xyzrgbL_ply(data_filename)
    else:
        print('Unknown file type! exiting.')
        exit()

    print('data load done')
    data = data[:, 0:6]
    data[:,3:6] /= 255.0
    idx = np.array(range(data.shape[0]))

    data_batch, idx_batch, isvalid, pts_num = scene2blocks(data, idx, num_point, block_size=block_size, inner_size=inner_size)

    #eigenvalues calculaition
    print('eigenvalues calculaiting...')
    idx = np.reshape(idx_batch, [-1])
    unique_idx = np.unique(idx)   
    eigs_batch = eig_cal_map(data[:,0:3], unique_idx, idx, radius)
    eigs_batch = np.reshape(eigs_batch, [-1, 6])
    data_batch = np.concatenate([data_batch, eigs_batch], -1)

    return data_batch, idx_batch, isvalid, pts_num



def scene2blocks(data, idx, num_point, block_size=1.0, inner_size=0.5, count_limit=100):
    """ Prepare block data for scene testing.
    DISCRIBTION:
    Each block consits of two part: the inner part with size: block_size*block_size*hight,
        and the buffer area around the inner part with side lenght buffer_size
    Args:
        data: N x 6 numpy array, 012 are XYZ in meters, 345 are RGB in [0,1]
        label_id: N size int numpy array
        block_size: float, physical size of the block in meters
        buffer_size: float, lenght of buffer area
    Returns:
        block_data_list: list, each element is the points selected from the block
        eachblock_pointnum_list: list, each element is the number of points selected from the block, [0, max_num_point]
        eachblock_validnum_list: list, each element is the number of points lying in the inner part
        block_label_id_list: list, each element is the label_idc of points lying in the inner part        
    """

    buffer_size = (block_size-inner_size)/2
    stride = inner_size

    data[:,0:2] -= np.amin(data, 0)[0:2]
    limit = np.amax(data, 0)[0:3]
     
    num_block_x = int(np.ceil(limit[0] / stride))
    num_block_y = int(np.ceil(limit[1] / stride))

    # Collect blocks
    block_data_list = []
    block_idx_list = []
    block_isvalid_list = []
    n_point_list = []

    for i in range(num_block_x):
        print('%d/%d'%(i, num_block_x))
        for j in range(num_block_y):
            xbeg = i*stride
            ybeg = j*stride
            
            #find points in [x-buffer,x+stride+buffer]*[y-buffer,y+stride+buffer]
            xcond = (data[:,0]<=xbeg+stride+buffer_size) & (data[:,0]>=xbeg-buffer_size)
            ycond = (data[:,1]<=ybeg+stride+buffer_size) & (data[:,1]>=ybeg-buffer_size)

            cond = xcond & ycond
            block_points = data[cond, :]
            block_idx = idx[cond]

            #block_points, block_idx, n_point = random_sample_data_label(block_points, block_idx, num_point)
            block_points, block_idx, n_point = FPS_data_label(block_points, block_idx, num_point, res=0.02)

            # then find points in [x,x+stride]*[y,y+stride]
            xcond = (block_points[:,0]<=xbeg+stride) & (block_points[:,0]>=xbeg)
            ycond = (block_points[:,1]<=ybeg+stride) & (block_points[:,1]>=ybeg)
            cond = xcond & ycond
            if np.sum(cond) < count_limit: # discard block if there are less than 50 pts in the inner part.
                continue
            
            #zero mean
            block_points[:,0:2] -= np.mean(block_points[:,0:2], 0)

            block_data_list.append(block_points)
            block_idx_list.append(block_idx)
            block_isvalid_list.append(cond)
            n_point_list.append(n_point)
    return np.concatenate(block_data_list, 0), np.concatenate(block_idx_list, 0), np.concatenate(block_isvalid_list, 0), np.array(n_point_list)


def FPS_data_label(data, label, num_sample, res=None):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]
    xyz = data[:,0:3]
    if (N > num_sample):
        if res is not None:
            ridx = downsampling(xyz, res)
            data = data[ridx,...]
            label = label[ridx,...]
            xyz = xyz[ridx,...]
        idx = FPS(xyz, num_sample)
        return data[idx, ...], label[idx, ...], num_sample
    else:
        return data, label, N

def downsampling(xyz, res):
    limitmax = np.amax(xyz, 0)[0:3]
    limitmin = np.amin(xyz, 0)[0:3]
    num_vox = np.ceil((limitmax-limitmin)/res)
    idx = np.ceil((xyz-limitmin)/res)
    idx = idx[:,0] + idx[:,1]*num_vox[0] + idx[:,2]*num_vox[0]*num_vox[1]
    uidx, ridx = np.unique(idx, return_index=True)
    return ridx


def random_sample_data_label(data, label, num_sample):
    """ data is in N x ...
        we want to keep num_samplexC of them.
        if N > num_sample, we will randomly keep num_sample of them.
        if N < num_sample, we will randomly duplicate samples.
    """
    N = data.shape[0]

    if (N > num_sample):
        sample = np.random.choice(N, num_sample)
        return data[sample, ...], label[sample, ...], num_sample
    else:
        return data, label, N


def read_xyzrgbL_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, red, green, blue, label] for x,y,z, red, green, blue, label in pc])
    return pc_array


# Write numpy array data and incice to h5_filename
def save_h5_data_and_idx(h5_filename, data, idx, mask, pts_num):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset('data', data=data, dtype='float32')
    h5_fout.create_dataset('idx', data=idx, dtype='int32')
    h5_fout.create_dataset('mask', data=mask, dtype='bool')
    h5_fout.create_dataset('pts_num', data=pts_num, dtype='int32')
    h5_fout.close()


if __name__=='__main__':
    # Note: the block size for testing is 3.6m*3.6m for speed consideration, which is larger than training 
    #settings
    org_datadir = '/media/wl/user/3D_pcseg_Data/S3DIS'

    indoor3d_data_dir = os.path.join(org_datadir, 'stanford_indoor3d')
    output_dir = os.path.join(ROOT_DIR, 'Data/S3DIS/test_dataset')
    NUM_POINT = 4096*9

    if not os.path.exists(output_dir): os.mkdir(output_dir)
    test_area = 'Area_5'
    
    Files = glob.glob(indoor3d_data_dir + '/*.ply')
    for i in range(len(Files)):
        data_filename = Files[i]
        if test_area in data_filename:
            h5_filename = os.path.join(output_dir, data_filename.split('.')[0].split('/')[-1]+'.h5')
            print(h5_filename)
            data, idx, mask, pts_num = test_area2blocks_zeromean(data_filename, NUM_POINT, block_size=3.6, inner_size=3.4, radius=0.1)
            save_h5_data_and_idx(h5_filename, data, idx, mask, pts_num)
    
