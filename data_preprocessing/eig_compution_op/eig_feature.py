from ctypes import *  
import numpy as np

# -----------------------------------------------------------------------------
# PREPARE BLOCK DATA FOR DEEPNETS TRAINING/TESTING
# -----------------------------------------------------------------------------
eigfeature_num=6

funs_c = CDLL("../eig_compution_op/build/libeig_com.so") 
def eig_compute(data, searchset, radius=0.15, k_nn=7):
    """
    input:
        data: (N,3)
        searchset: (M, 3)
    return:
        eig: (M,6)
    """
    
    point_num = data.shape[0]
    data = np.reshape(data, [-1])
    search_num = searchset.shape[0]
    searchset = np.reshape(searchset, [-1])  
    data = (c_float*(point_num*3))(*data)
    searchset = (c_float*(search_num*3))(*searchset)

    eig_inf = (c_float*(search_num*eigfeature_num))()

    point_num_c = (c_int*1)(point_num)
    search_num_c = (c_int*1)(search_num)
    radius_c = (c_float*1)(radius)
    k_nn_c = (c_int*1)(k_nn)

    funs_c.compute_eigen(data, point_num_c, searchset, search_num_c, radius_c, k_nn_c, eig_inf)

    return np.reshape(eig_inf, [search_num,-1])


def eig_mapping(eig_inf, unique_idx, idx):
    """
    input:
        unique_idx: (M)
        idx: (N)
        eigs: (M, 3)
    return:
        eigs_batch: (N,6)
    """

    sampled_num = len(idx)
    unique_num = len(unique_idx)

    eig_inf = np.reshape(eig_inf, [-1])
    unique_idx = np.reshape(unique_idx, [-1])
    idx = np.reshape(idx, [-1])
    eig_inf = (c_float*(unique_num*eigfeature_num))(*eig_inf)
    unique_idx = (c_int*unique_num)(*unique_idx)
    idx = (c_int*sampled_num)(*idx)

    eigs_batch = (c_float*(sampled_num*eigfeature_num))()

    unique_num_c = (c_int*1)(unique_num)
    sampled_num_c = (c_int*1)(sampled_num)

    print('eigs mapping...')
    funs_c.eiginf_mapping(eig_inf, eigs_batch, unique_idx, unique_num_c, idx, sampled_num_c)

    return np.reshape(eigs_batch, [sampled_num,-1])


def eig_cal_map(data, unique_idx, idx, radius=0.15, k_nn=21):
    """
    input:
        data: (N, 3)
        unique_idx: (M)
        idx: (N)
    return:
        eigs_batch: (N,3)
    """
    point_num = data.shape[0]
    searchset = data[unique_idx, :]
    search_num = searchset.shape[0]
    sampled_num = len(idx)

    data = np.reshape(data, [-1])
    searchset = np.reshape(searchset, [-1])  
    unique_idx = np.reshape(unique_idx, [-1])
    idx = np.reshape(idx, [-1])
    data = (c_float*(point_num*3))(*data)
    searchset = (c_float*(search_num*3))(*searchset)
    unique_idx = (c_int*search_num)(*unique_idx)
    idx = (c_int*sampled_num)(*idx)

    eigs_batch = (c_float*(sampled_num*eigfeature_num))()

    point_num_c = (c_int*1)(point_num)
    search_num_c = (c_int*1)(search_num)
    sampled_num_c = (c_int*1)(sampled_num)
    radius_c = (c_float*1)(radius)
    k_nn_c = (c_int*1)(k_nn)

    funs_c.eig_cal_map(data, point_num_c, searchset, search_num_c, unique_idx, idx, sampled_num_c, radius_c, k_nn_c, eigs_batch)

    return np.reshape(eigs_batch, [sampled_num,-1])


def eig_cal_map_intensity(data, intensity, unique_idx, idx, radius=0.15, k_nn=7):
    """
    input:
        data: (N, 3)
        intensity: (N)
        unique_idx: (M)
        idx: (N)
    return:
        eigs_batch: (N,4)
    """
    point_num = data.shape[0]
    searchset = data[unique_idx, :]
    search_num = searchset.shape[0]
    sampled_num = len(idx)

    data = np.reshape(data, [-1])
    searchset = np.reshape(searchset, [-1])  
    unique_idx = np.reshape(unique_idx, [-1])
    idx = np.reshape(idx, [-1])
    intensity = np.reshape(intensity, [-1])
    data = (c_float*(point_num*3))(*data)
    searchset = (c_float*(search_num*3))(*searchset)
    unique_idx = (c_int*search_num)(*unique_idx)
    idx = (c_int*sampled_num)(*idx)
    intensity = (c_float*point_num)(*intensity)

    eigs_batch = (c_float*(sampled_num*4))()

    point_num_c = (c_int*1)(point_num)
    search_num_c = (c_int*1)(search_num)
    sampled_num_c = (c_int*1)(sampled_num)
    radius_c = (c_float*1)(radius)
    k_nn_c = (c_int*1)(k_nn)

    funs_c.eig_cal_map_intensity(data, point_num_c, intensity, searchset, search_num_c, unique_idx, idx, sampled_num_c, radius_c, k_nn_c, eigs_batch)

    return np.reshape(eigs_batch, [sampled_num,-1])



def scene2block_C(pts_xyz, MAX_NUM, block_size, count_threshold):

    '''
        pts_xyz: (pts_num, 3)
    '''

    pts_num, channel = pts_xyz.shape
    assert(channel==3)
    limit = np.amax(pts_xyz, 0)[0:3]
    x_num = int( np.ceil(limit[0] / block_size) )
    y_num = int( np.ceil(limit[1] / block_size) )
    total_block_num = x_num*y_num

    pts_xyz = np.reshape(pts_xyz, [-1])
    pts_xyz = (c_float*(pts_num*3))(*pts_xyz)
    sample_idx = (c_int*(total_block_num*MAX_NUM))()

    pts_num = (c_int*1)(pts_num)
    MAX_NUM_c = (c_int*1)(MAX_NUM)
    x_num = (c_int*1)(x_num)
    count_threshold = (c_int*1)(count_threshold)
    block_num = (c_int*1)()
    block_size = (c_float*1)(block_size)

    funs_c.scene2blocks(pts_xyz, sample_idx, MAX_NUM_c, pts_num, x_num, block_size, count_threshold, block_num)

    block_num = np.reshape(block_num, [-1])[0]
    sample_idx = np.reshape(sample_idx, [-1])
    sample_idx = sample_idx[0:block_num*MAX_NUM]

    return sample_idx, block_num


def scene2blocks_withinner(pts_xyz, MAX_NUM, block_size, inner_size, count_threshold):

    '''
        pts_xyz: (pts_num, 3)
    '''

    pts_num, channel = pts_xyz.shape
    assert(channel==3)
    limit = np.amax(pts_xyz, 0)[0:2]
    x_num = int( np.ceil(limit[0] / block_size) )
    y_num = int( np.ceil(limit[1] / block_size) )
    total_block_num = x_num*y_num

    inner_radius = inner_size/2

    pts_xyz = np.reshape(pts_xyz, [-1])
    pts_xyz = (c_float*(pts_num*3))(*pts_xyz)
    sample_idx = (c_int*(total_block_num*MAX_NUM))()
    isvalid = (c_float*(total_block_num*MAX_NUM))()

    pts_num = (c_int*1)(pts_num)
    MAX_NUM_c = (c_int*1)(MAX_NUM)
    x_num = (c_int*1)(x_num)
    count_threshold = (c_int*1)(count_threshold)
    block_num = (c_int*1)()
    block_size = (c_float*1)(block_size)
    inner_radius = (c_float*1)(inner_radius)

    funs_c.scene2blocks_withinner(pts_xyz, sample_idx, isvalid, MAX_NUM_c, pts_num, x_num, block_size, inner_radius, count_threshold, block_num)

    block_num = np.reshape(block_num, [-1])[0]
    sample_idx = np.reshape(sample_idx, [-1])
    isvalid = np.reshape(isvalid, [-1])

    sample_idx = sample_idx[0:block_num*MAX_NUM]
    isvalid = isvalid[0:block_num*MAX_NUM]

    return sample_idx, isvalid, block_num


def FPS(pts_xyz, MAX_NUM):

    '''
        pts_xyz: (pts_num, 3)
    '''
    pts_num, channel = pts_xyz.shape
    assert(channel==3)

    pts_xyz = np.reshape(pts_xyz, [-1])
    pts_xyz = (c_float*(pts_num*3))(*pts_xyz)
    sample_idx = (c_int*MAX_NUM)()

    #batch_num = (c_int*1)(1)
    pts_num = (c_int*1)(pts_num)
    MAX_NUM_c = (c_int*1)(MAX_NUM)
    
    funs_c.FPS_pythonwarpper(pts_num, MAX_NUM_c, pts_xyz, sample_idx)

    sample_idx = np.reshape(sample_idx, [-1])

    return sample_idx

