import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
tf_op_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_op_so.so'))

def query_ball_pnet_point(radius, nsample, xyz1, xyz2):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    #return grouping_module.query_ball_point(radius, nsample, xyz1, xyz2)
    return tf_op_module.query_ball_pnet_point(xyz1, xyz2, radius, nsample)
ops.NoGradient('QueryBallPnetPoint')

def rand_seeds(inp):
    '''
    input:
        batch_size * ndataset * 3   float32
    returns:
        batch_size          int32
    '''
    return tf_op_module.rand_seeds(inp)
ops.NoGradient('RandSeeds')

def shuffle_ids(inp):
    '''
input:
    batch_size * ndataset * 3   float32
returns:
    batch_size  *   ndataset       int32
    '''
    return tf_op_module.shuffle_ids(inp)
ops.NoGradient('ShuffleIds')

def FPS_downsample(inp, stride):
    '''
    input:
        batch_size * ndataset * 3   float32
        int:  downsampling factor
    returns:
        batch_size * npoint         int32

    '''
    seed = rand_seeds(inp)
    return tf_op_module.farthest_point_sample(inp, seed, stride)
ops.NoGradient('FarthestPointSample')


def query_ball_knn(data_xyz, search_xyz, radius, nsample):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        data_xyz: (batch_size, ndataset, 3) float32 array, input points
        search_xyz: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    
    with tf.device('/cpu:0'):
        shuffled_ids = shuffle_ids(data_xyz) 
    return tf_op_module.query_ball_knn(data_xyz, search_xyz, shuffled_ids, radius, nsample)
ops.NoGradient('QueryBallKnn')


def query_ball(data_xyz, search_xyz, radius, nsample):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        data_xyz: (batch_size, ndataset, 3) float32 array, input points
        search_xyz: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    with tf.device('/cpu:0'):
        shuffled_ids = shuffle_ids(data_xyz)
    return tf_op_module.query_ball(data_xyz, search_xyz, shuffled_ids, radius, nsample)
ops.NoGradient('QueryBall')

def knn_search(data_xyz, search_xyz, k_num):
    '''
    Input:
        k_num: int32, number of k nearst points
        data_xyz: (batch_size, nsearchset, 3) float32 array, search points
        search_xyz: (batch_size, ndataset, 3) float32 array, dataset points
    Output:
        idx: (batch_size, ndataset, k_num) int32 array, indices of knn points
        dist: (batch_size, ndataset, k_num) float32 array, distance of knn points (without sort)
    '''
    return tf_op_module.knn_search(data_xyz, search_xyz, k_num)
ops.NoGradient('KnnSearch')

def gather_point(inp,idx):
    '''
input:
    batch_size * ndataset * 3   float32
    batch_size * npoints        int32
returns:
    batch_size * npoints * 3    float32
    '''
    return tf_op_module.gather_point(inp,idx)

@tf.RegisterGradient('GatherPoint')
def _gather_point_grad(op,out_g):
    inp=op.inputs[0]
    idx=op.inputs[1]
    return [tf_op_module.gather_point_grad(inp,idx,out_g),None]

def gather_id(inp,idx):
    '''
input:
    batch_size * ndataset * 3   int32
    batch_size * npoints        int32
returns:
    batch_size * npoints * 3    int32
    '''
    return tf_op_module.gather_id(inp,idx)
ops.NoGradient('GatherId')

def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return tf_op_module.group_point(points, idx)
@tf.RegisterGradient('GroupPoint')
def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [tf_op_module.group_point_grad(points, idx, grad_out), None]

def gather_eigvector(eigvs, idx):
    '''
    Input:
        points: (batch_size, ndataset, 3,3) float32 array, points to sample from
        idx: (batch_size, ndataset) int32 array, indices to points
    Output:
        out: (batch_size, ndataset, 3) float32 array, values sampled from points
    '''
    return tf_op_module.gather_eigvector(eigvs, idx)
@tf.RegisterGradient('GatherEigvector')
def _gather_eigvector_grad(op, grad_out):
    eigvs = op.inputs[0]
    idx = op.inputs[1]
    return [tf_op_module.gather_eigvector_grad(eigvs, idx, grad_out), None]


def three_interpolate(points, idx, weight):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    return tf_op_module.three_interpolate(points, idx, weight)
@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [tf_op_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]
