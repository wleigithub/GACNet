
import os, pdb
import sys
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
sys.path.append(os.path.join(BASE_DIR, '../tf_ops'))
import tensorflow as tf
import numpy as np
import tf_util
from pcnet_util import build_graph, graph_coarse, graph_attention_layer, graph_pooling_layer, point_upsample_layer, crf_layer, graph_attention_layer_for_featurerefine
from tf_ops import knn_search


def placeholder_inputs(feature_channel, with_spw=False):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(None, None, feature_channel))
    labels_pl = tf.placeholder(tf.int32, shape=(None, None))
    is_training_pl = tf.placeholder(tf.bool, shape=())
    if with_spw:
        spws_pl = tf.placeholder(tf.float32, shape=(None, None))
        return pointclouds_pl, labels_pl, spws_pl, is_training_pl
    else:
        return pointclouds_pl, labels_pl, is_training_pl


def build_graph_pyramid(xyz, graph_inf):
    """ Builds a pyramid of graphs and pooling operations corresponding to progressively coarsened point cloud.
    Inputs:
        xyz: (batchsize, num_point, nfeature)
        graph_inf: parameters for graph building (see run.py)
    Outputs:
        graph_prd: graph pyramid contains the vertices and their edges at each layer
        coarse_map: record the corresponding relation between two close graph layers (for graph coarseing/pooling)
    """
    stride_list, radius_list, maxsample_list = graph_inf['stride_list'], graph_inf['radius_list'], graph_inf['maxsample_list']

    graph_prd = []
    graph = {} #save subsampled points and their neighbor indexes at each level
    coarse_map = [] # save index map from previous level to next level

    ids = build_graph(xyz, radius_list[0], maxsample_list[0]) # (batchsize, num_point, maxsample_list[0]) neighbor indexes at current level 
    graph['vertex'], graph['adjids'] = xyz, ids 
    graph_prd.append(graph.copy())

    for stride, radius, maxsample in zip(stride_list, radius_list[1:], maxsample_list[1:]):
        xyz, coarse_map_ids = graph_coarse(xyz, ids, stride, radius,  maxsample)
        coarse_map.append(coarse_map_ids)
        ids = build_graph(xyz, radius, maxsample)
        graph['vertex'], graph['adjids'] = xyz, ids 
        graph_prd.append(graph.copy())

    return graph_prd, coarse_map


def build_network(graph_prd, coarse_map, net_inf, features, num_class, is_training, bn_decay=None):
    """build GACNet on the graph pyramid.
    Inputs:
        graph_prd: graph pyramid contains the vertices and their edges at each layer
        coarse_map: record the corresponding relation between two close graph layers (for graph coarseing/pooling)
        net_inf: parameters for GACNet (see run.py for details) 
        features: input signal (x,y,z,r,g,b,...)
        num_class: number of class to be classified
        is_training: training or not
        bn_decay: use weight decay
    Outputs:
        features: learned feature for each point
    """
    forward_parm, upsample_parm, fullconect_parm = net_inf['forward_parm'], net_inf['upsample_parm'], net_inf['fullconect_parm']

    inif = features[...,0:6] # (x,y,z,r,g,b)
    features = features[...,2:] #(z, r, g, b, and (initial geofeatures if possible))
    #features = tf.concat( [features[...,2:3],features[...,6:]], axis=-1) #remove the rgb

    #forward layers
    feature_prd = [] 
    for i in range(len(coarse_map)):
        features = graph_attention_layer(graph_prd[i], features, forward_parm[i][0], forward_parm[i][1], is_training, bn_decay, scope='attention_%d'%(i), bn=True)
        feature_prd.append(features)
        features = graph_pooling_layer(features, coarse_map[i], scope='graph_pooling_%d'%(i))

    features = graph_attention_layer(graph_prd[-1], features, forward_parm[-1][0], forward_parm[-1][1], is_training, bn_decay, scope='attention_%d'%(len(coarse_map)), bn=True)   
    
    #upsample/interpolation layers
    for i in range(len(coarse_map)):
        j = len(coarse_map) -i-1
        features = point_upsample_layer(graph_prd[j]['vertex'], graph_prd[j+1]['vertex'], feature_prd[j], features, 
                                        upsampling=True, mlp=upsample_parm[j], is_training=is_training, bn_decay=bn_decay, scope='up%d'%(j))

    #fully connected layer
    features = tf_util.conv1d(features, fullconect_parm, 1,  bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    features = tf_util.dropout(features, keep_prob=0.5, is_training=is_training, scope='dp1')
    features = tf_util.conv1d(features, num_class, 1, activation_fn=None, is_training=is_training, scope='fc2')

    #features = crf_layer(graph_prd[0]['vertex'], inif[...,3:6], features, graph_prd[0]['adjids'], is_training, bn_decay, scope='crflayer', iter_num=5)
    features = graph_attention_layer_for_featurerefine(inif, features, graph_prd[0]['adjids'], is_training, bn_decay, scope='refine') # in this layer, the ouput is not through actication function

    return features


def get_loss(pred, label, spw=None):
    """ 
        pred: BxNxC,
        label: BxN, 
	    smpw: BxN 
    """
    if spw is not None:
        classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred, weights=spw)
    else:
        classify_loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=pred)
    tf.summary.scalar('classify loss', classify_loss)
    return classify_loss


# test ------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    import pdb
    num_point = 4096
    data = np.random.rand(10, num_point, 6)
    #xyz = tf.constant(data, dtype=np.float32)
    xyz = tf.placeholder(tf.float32, shape=(None,None,6))
    num_point = tf.placeholder(tf.int32,shape=())

    graph_inf = {'stride_list': [4, 4, 4, 2],
                'radius_list': [0.1, 0.2, 0.4, 0.8, 1.6],
                'maxsample_list': [12, 21, 21, 21, 12]
    }

    forward_parm = [
                    [ [32,32,64], [64] ],
                    [ [64,64,128], [128] ],
                    [ [128,128,256], [256] ],
                    [ [256,256,512], [512] ],
                    [ [256,256], [256] ]
    ]
    upsample_parm = [
                    [128, 128],
                    [128, 128],
                    [256, 256],
                    [256, 256]
    ]
    fullconect_parm = 128

    net_inf = {'forward_parm': forward_parm,
            'upsample_parm': upsample_parm,
            'fullconect_parm': fullconect_parm
    }

    #with tf.Graph().as_default():
    graph_prd, coarse_map = build_graph_pyramid(xyz, graph_inf)
    logits = build_network(graph_prd, coarse_map, net_inf, xyz, 10, tf.constant(True))

    #pdb.set_trace()