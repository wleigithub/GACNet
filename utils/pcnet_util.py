
import tensorflow as tf
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops'))

from tf_ops import FPS_downsample, query_ball_knn, knn_search, gather_point, gather_id, group_point, three_interpolate, query_ball, gather_eigvector

import tf_util
import pdb

#parameters for graph attention convolution (GAC)
#which combines both the position and feature difference to assign weights to different neighbors
gac_par = [
    [32, 16], #MLP for xyz 
    [16, 16], #MLP for feature 
    [64] #hidden node of MLP for mergering 
]

##############################################################################################################################
#------------------------------------------graph building modules------------------------------------------------------------
##############################################################################################################################
def build_graph(xyz, radius,  maxsample):
    """ Converts point cloud to graph.
    Inputs:
        xyz: (batchsize, npoint, nfeature)
        radius: radius to search neighbors
        maxsample: max number to sample in their neighbors
    Outputs:
        ids: neighbors' indices
    """
    ids, cnts = query_ball(xyz, xyz, radius, maxsample) #(batchsize, npoint, maxsample) (batchsize, npoint, maxsample)
    return ids         
    

def graph_coarse(xyz_org, ids_full, stride, radius,  maxsample):
    """ Coarse graph with down sampling, and find their corresponding vertexes at previous (or father) level. """
    if stride>1:
        sub_pts_ids = FPS_downsample(xyz_org, stride) #(batch_size, num_point)
        sub_xyz = gather_point(xyz_org, sub_pts_ids) #(batch_size, num_point, 3)
        ids = gather_id(ids_full, sub_pts_ids)#(batchsize, num_point, maxsample)
        return sub_xyz, ids
    else:
        return xyz_org, ids_full
    

##############################################################################################################################
#------------------------------------------graph network modules------------------------------------------------------------
##############################################################################################################################

def covInf(grouped_xyz):
    #calculate covMat
    pts_mean = grouped_xyz - tf.reduce_mean(grouped_xyz, axis=2, keep_dims=True)  # (b, n, K, 3)
    pts_BNK31 = tf.expand_dims(pts_mean, axis=-1)
    covMat = tf.matmul(pts_BNK31, pts_BNK31, transpose_b=True)  # (b, n, K, 3, 3)
    covMat = tf.reduce_mean(covMat, axis=2, keep_dims=False)  # (b, n, 3, 3)

    covMat_flat = tf.concat([covMat[...,0],covMat[...,1],covMat[...,2]], axis=-1)
    '''
    #calculate covInf
    eigs, eigvs = tf.self_adjoint_eig(covMat)
    eigs = tf.nn.l2_normalize(tf.abs(eigs))
    eigs, idx = tf.nn.top_k(eigs, k=3)
    #
    idx = idx[...,2]
    #pdb.set_trace()
    eigv = gather_eigvector(eigvs, idx)
    eigv = tf.nn.l2_normalize(eigv)
    
    return tf.concat([eigs, eigv], axis=-1)
    '''
    return covMat_flat


def coeff_generation(grouped_features, features, grouped_xyz, is_training, bn_decay, scope, bn=True, mode='with_feature'):
    with tf.variable_scope(scope) as sc:
        if mode == 'with_feature':
            coeff = grouped_features - tf.expand_dims(features, axis=2)
            coeff = tf_util.MLP_2d(coeff, gac_par[1], is_training, bn_decay, bn, scope='conv_with_feature')## compress to a hiden feature space  
            coeff = tf.concat([grouped_xyz, coeff], -1)
        if mode == 'edge_only':
            coeff = grouped_xyz
        if mode == 'feature_only':
            coeff = grouped_features - tf.expand_dims(features, axis=2)
            coeff = tf_util.MLP_2d(coeff, gac_par[1], is_training, bn_decay, bn, scope='conv_feature')## compress to a hiden feature space 

        grouped_features = tf.concat([grouped_xyz, grouped_features], axis=-1) #updata feature
        out_chal = grouped_features.get_shape()[-1].value
        coeff = tf_util.MLP_2d(coeff, gac_par[2], is_training, bn_decay, bn, scope='conv')## map to a hiden feature space     
        coeff = tf_util.conv2d(coeff, out_chal, [1,1], scope='conv2d', is_training=is_training, bn_decay=bn_decay, activation_fn=None)  #output coefficent
        coeff = tf.nn.softmax(coeff, axis=2) #coefffient normalization

        grouped_features = tf.multiply(coeff, grouped_features)
        grouped_features = tf.reduce_sum(grouped_features, axis=[2], keep_dims=False)
        return grouped_features


def graph_attention_layer(graph, features, mlp, mlp2, is_training, bn_decay, scope, bn=True):
    with tf.variable_scope(scope) as sc:
        xyz, ids = graph['vertex'], graph['adjids']
        grouped_xyz = group_point(xyz, ids) # (batch_size, ndataset, maxsample, 3)
        '''
        #it's slow to calculate the inital features during training, so we compute it prior
        #calculate inital features
        if int(scope.split('_')[1])==0:
            cov_inf = covInf(grouped_xyz)
            features = tf.concat([features, cov_inf], axis=-1)
        '''
        grouped_xyz -= tf.expand_dims(xyz, 2) # translation normalization
        grouped_xyz = edge_mapping(grouped_xyz, is_training, bn_decay, scope='edge') # map local postion to a feature space with MLP
        features = tf_util.MLP_1d(features, mlp, is_training, bn_decay, bn, scope='feature') #feature transform
        grouped_features = group_point(features, ids) # (batch_size, ndataset, maxsample, channel)

        new_features = coeff_generation(grouped_features, features,grouped_xyz, is_training, bn_decay, scope='coeff_gen')
        #new_features = tf.reduce_max(tf.concat([grouped_xyz, grouped_features], axis=-1), axis=2, keep_dims=False)  #for comparison of GAC and MAX
        #new_features = tf.reduce_mean(tf.concat([grouped_xyz, grouped_features], axis=-1), axis=2, keep_dims=False)  #for comparison of GAC and MEAN
        
        if mlp2 is not None and features is not None:
            new_features = tf.concat([features, new_features], axis=-1) # (batch_size, ndataset, maxsample, 3+channel) #merge the feature with itself
            new_features = tf_util.MLP_1d(new_features, mlp2, is_training, bn_decay, bn)
        return new_features


def graph_attention_layer_for_featurerefine(initf, features, ids, is_training, bn_decay, scope, bn=True):
    """
    Graph attention convolution for post-processing  (for comparion convenience with CRF)
    """
    with tf.variable_scope(scope) as sc:
        out_chal = features.get_shape()[-1].value

        grouped_initf = group_point(initf, ids) # (batch_size, ndataset, maxsample, 3)
        grouped_initf -= tf.expand_dims(initf, 2) # translation normalization
        grouped_initf = edge_mapping(grouped_initf, is_training, bn_decay, scope='edge') #map local postion to a feature space with MLP
        grouped_features = group_point(features, ids) # (batch_size, ndataset, maxsample, channel)

        new_features = coeff_generation(grouped_features, features, grouped_initf, is_training, bn_decay, scope='coeff_gen')
        #new_features = tf.reduce_max(tf.concat([grouped_initf, grouped_features], axis=-1), axis=2, keep_dims=False)

        new_features = tf.concat([features, new_features], axis=-1) # (batch_size, ndataset, maxsample, 3+channel) #merge the feature with itself
        new_features = tf_util.conv1d(new_features, out_chal, 1, activation_fn=None, is_training=is_training, scope=scope)
        return new_features


def edge_mapping(grouped_xyz, is_training, bn_decay, scope, bn=True):
    """ mapping edges (deta.x) into feature space {deep sets}
    input: grouped_xyz (b,n,k,3)
    output: (b,n,k,output_channel)
    """
    with tf.variable_scope(scope) as sc:
        #MLP for edge feature transform
        grouped_xyz = tf_util.MLP_2d(grouped_xyz, gac_par[0], is_training, bn_decay, bn, scope='pn_conv')
        return grouped_xyz


##############################################################################################################################

def graph_pooling_layer(features, coarse_map, scope, pooling='max'):
    ''' 
        Input:                                                                                                      
            xyz: (batch_size, ndataset, 3) TF tensor                                                                                                       
            features: (batch_size, ndataset, nchannel) TF tensor   
            ball_ids: (batch_size, ndataset, maxsample) TF tensor
            num_subsampling: int32 or None, None means no subsampling                                                
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_xyz: (batch_size, num_subsampling or ndataset, 3) TF tensor  
            new_features: (batch_size, num_subsampling or ndataset, mlp[-1]) TF tensor
    ''' 
    with tf.variable_scope(scope) as sc:
        grouped_features = group_point(features, coarse_map) # (batch_size, ndataset or num_subsampling, nsample, channel)

        if pooling=='max':
            new_features = tf.reduce_max(grouped_features, axis=[2]) # (batch_size,  ndataset or num_subsampling, channel)

        return new_features


def point_upsample_layer(xyz1, xyz2, features1, features2, upsampling, mlp, is_training, bn_decay, scope, bn=True):
    ''' 
        Input:                                                                                                      
            xyz1: (batch_size, ndataset1, 3) TF tensor                                                              
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1                                           
            features1: (batch_size, ndataset1, nchannel1) TF tensor                                                   
            features2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point                                                 
        Return:
            new_features: (batch_size, ndataset1, mlp[-1]) TF tensor
    ''' 
    with tf.variable_scope(scope) as sc:
        if upsampling:
            idx, dist = knn_search(data_xyz=xyz2, search_xyz=xyz1, k_num=3)
            dist = 1.0/tf.maximum(dist, 1e-10)
            weight = tf.nn.softmax(dist, axis=2)
            #with tf.device('/cpu:0'):
            interpolated_features = three_interpolate(features2, idx, weight)
        else:
            interpolated_features = features2
            
        new_features = tf.concat(axis=2, values=[interpolated_features, features1]) # B,ndataset1,nchannel1+nchannel2
        
        if mlp is not None:
            new_features = tf_util.MLP_1d(new_features, mlp, is_training, bn_decay, bn)

        return new_features

##############################################################################################################################
#------------------------------------------crf modules------------------------------------------------------------
##############################################################################################################################

def crf_layer(xyz, rgb, features, idx, is_training, bn_decay, scope, iter_num=1):
    """
    CRF module with Gaussian kernel as post-processing
    """
    with tf.variable_scope(scope) as sc:
        #idx, dist = knn_search(xyz, xyz, knn_num)
        grouped_xyz = group_point(xyz, idx) # (batch_size, num_point, knn_num, 3)
        grouped_xyz -= tf.expand_dims(xyz, 2)
        grouped_rgb = group_point(rgb, idx) # (batch_size, num_point, knn_num, 3)
        grouped_rgb -= tf.expand_dims(rgb, 2) # translation normalization
        
        #Gaussian weights
        theta = tf.Variable(tf.truncated_normal([3], mean=1, stddev=0.01))  #(theta_alpha, theta_beta, theta_gama)
        W = tf.Variable(tf.truncated_normal([2], mean=0.5, stddev=0.01))

        num_class = features.get_shape()[-1].value
        compatible_weight = tf.get_variable('compatible_weight', shape=[1, num_class, num_class], initializer=tf.constant_initializer(np.identity(num_class)))

        #initial normlizing
        #features = tf.nn.softmax(features)
        for i in range(iter_num):
            features_normed = tf.nn.softmax(features)
            grouped_features = group_point(features_normed, idx) #(batch_size, num_point, knn_num, channels)

            #compute weights with Gaussian kernels
            ker_appearance = tf.reduce_sum(tf.square(grouped_xyz*theta[0]), axis=3) + tf.reduce_sum(tf.square(grouped_rgb*theta[1]), axis=3) #(batch_size, num_point, knn_num)
            ker_smooth = tf.reduce_sum(tf.square(grouped_xyz*theta[2]), axis=3)
            ker_appearance = tf.exp(-ker_appearance)
            ker_smooth = tf.exp(-ker_smooth)

            Q_weight = W[0]*ker_appearance + W[1]*ker_smooth  #(batch_size, num_point, knn_num)
            Q_weight = tf.expand_dims(Q_weight, axis=2)  #(batch_size, num_point, 1, knn_num)

            #message passing
            Q_til_weighted = tf.matmul(Q_weight, grouped_features) #(batch_size, num_point, 1, channels)
            Q_til_weighted = tf.squeeze(Q_til_weighted, axis=2) #(batch_size, num_point, channels)

            #compatibility transform
            Q_til_weighted = tf.nn.conv1d(Q_til_weighted, compatible_weight, 1, padding='SAME')

            #adding unary potentials
            features += Q_til_weighted

            #normalization
            #features = tf.nn.softmax(features)
    return features
