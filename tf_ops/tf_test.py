import tensorflow as tf
import numpy as np
import random
from tf_ops import knn_search, three_interpolate, group_point, gather_point, FPS_downsample, query_ball_knn, shuffle_ids,query_ball
import pdb


if __name__=='__main__':
    #arr_xyz1 = np.random.random((1,128,3)).astype('float32')
    arr_xyz2 = np.random.random((1,68,3)).astype('float32')
    arr_xyz1 = np.loadtxt('/media/wl/data/CVPR/pointnet-master/data/modelnet40_ply_hdf5_2048/test/1.txt')
    arr_xyz1 = np.reshape(arr_xyz1[:, 0:3], [1,-1, 3])
    #pdb.set_trace()
    xyz1 = tf.constant(arr_xyz1, dtype=tf.float32)
    xyz2 = tf.constant(arr_xyz1[:,0:500,:], dtype=tf.float32)
    #idxs, dist = knn_search(xyz2, xyz1, 3)
    #idxs, dist = query_ball_knn(xyz1, xyz1, 0.2, 15)
    #idxs, dist = query_ball(xyz2, xyz1, 0.1, 21)
    idxs = FPS_downsample(xyz1, 3)
    
    #idxs = shuffle_ids(xyz1)
    #pts = gather_point(xyz1, idxs)
    #with tf.device('/cpu:0'):
    #    shuffled_ids = shuffle_ids(xyz1)
    with tf.Session() as sess:  
        idx_out = sess.run(idxs) 
    idx= tf.constant(idx_out)
    pdb.set_trace()
