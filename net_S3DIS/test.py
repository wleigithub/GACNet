import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import glob
import pdb

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
sys.path.append(os.path.join(ROOT_DIR, 'test_utils'))

from three_interpolate import three_interpolate
import provider
from model import *

def test(args, graph_inf, net_inf):
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(args.gpu)):
            pointclouds_pl, labels_pl, is_training_pl = placeholder_inputs(args.channel)

            # Get model and loss 
            graph_prd, coarse_map = build_graph_pyramid(pointclouds_pl[...,0:3], graph_inf)
            logits = build_network(graph_prd, coarse_map, net_inf, pointclouds_pl, args.num_class, is_training_pl, bn_decay=None)
            pred = tf.nn.softmax(logits)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = True
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:False})

        last_model = tf.train.latest_checkpoint(args.log_dir)
        saver.restore(sess, last_model)
        ops = {'pointclouds_pl': pointclouds_pl,
               #'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred' : pred
               }

        TEST_FILES = glob.glob(args.testdataset_dir + '/*.h5')
        for i in range(len(TEST_FILES)):
            file_name = TEST_FILES[i]
            print(file_name+'\n')
            testset = h5py.File(file_name)
            points = testset['data'][:] # (blocks_num, 4096, 12)
            idx = testset['idx'][:]
            mask = testset['mask'][:]
            pts_num = testset['pts_num'][:]
            
            probs = []
            start_id =0
            end_id =0
            for j in range(len(pts_num)):
                end_id += pts_num[j]
                current_data = points[start_id:end_id,:]
                current_data = np.expand_dims(current_data, 0)
                feed_dict = {ops['pointclouds_pl']: current_data,  ops['is_training_pl']: False}
                current_probs = sess.run(ops['pred'], feed_dict=feed_dict)
                probs.append(np.squeeze(current_probs, 0))
                start_id += pts_num[j]
            probs = np.concatenate(probs, 0)
            probs = probs[mask,:]
            idx = idx[mask]

            out_filename = file_name.split('/')[-1].split('.')[0]
            save_h5_probs_idx(os.path.join(args.outdir, out_filename+'.h5'), probs, idx)

        sess.close()


def interpolate(args):
    """
    Calculate the label for the orignal point cloud
        Since we only predict the label of part of the points sampled from the orginal point cloud, 
        we have to calculate the label of the orignal point cloud with interpolation algorithm.
        Similar as we do for feature interpolation in the network, we also interpolate the label of each point (of the orignal point cloud) according to its three nearest neighbors
    """
    tmp_files = glob.glob(os.path.join(args.outdir, '*.h5'))
    for i in range(len(tmp_files)):
        file_name = tmp_files[i]
        filename = file_name.split('/')[-1].split('.')[0]
        print(filename)

        probs, idx = load_h5_probs_idx(os.path.join(args.outdir, filename+'.h5'))
        org_pts = provider.read_xyzrgbL_ply( os.path.join(args.test_dir, filename+'.ply') )
        true_label = org_pts[:,-1]
        probs_whole = three_interpolate(org_pts, idx, probs, args.num_class)
        pred_label = np.argmax(probs_whole, axis=-1)
        pred_label = np.uint8(pred_label) 

        save_labels_pred_true(os.path.join(args.outdir, filename+'_labels.pred_true'), pred_label, true_label)

        # only save the points without ceiling for vision efficiency
        idx_without_ceiling = np.where(true_label!=0)[0]
        pts_xyz = org_pts[idx_without_ceiling, 0:3]
        pred_label = pred_label[idx_without_ceiling]
        #provider.write_xyzL_ply(pts_xyz, pred_label, os.path.join(args.outdir, filename+'.ply'))

        os.remove(os.path.join(args.outdir, filename+'.h5'))


############################################################################################################
#---------------------------- utilized functions
############################################################################################################
# Write numpy array data and label to h5_filename
def save_h5_probs_idx(h5_filename, probs, idx, data_dtype='float32', idx_dtype='int32'):
    h5_fout = h5py.File(h5_filename)
    h5_fout.create_dataset(
            'probs', data=probs,
            compression='gzip', compression_opts=4,
            dtype=data_dtype)
    h5_fout.create_dataset(
            'idx', data=idx,
            compression='gzip', compression_opts=1,
            dtype=idx_dtype)
    h5_fout.close()


def load_h5_probs_idx(h5_filename):
    f = h5py.File(h5_filename)
    probs = f['probs'][:]
    idx = f['idx'][:]
    return probs, idx


def save_labels_pred_true(filename, pred_label, true_label):
    f = open(filename, 'w')
    for i in range(len(true_label)):
        f.write('%f %f\n' % (pred_label[i], true_label[i])) 
    f.close()


def compute_iou2(args, predl, truel):
    gt_classes = [0 for _ in range(args.num_class)]
    positive_classes = [0 for _ in range(args.num_class)]
    true_positive_classes = [0 for _ in range(args.num_class)]
    for i in range(len(truel)):
        gt_classes[truel[i]] += 1
        positive_classes[predl[i]] += 1
        true_positive_classes[truel[i]] += int(truel[i]==predl[i])
    
    OA = sum(true_positive_classes)/float(len(truel))
    iou_list = []
    for i in range(args.num_class):
        iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i] + 1e-9) 
        iou_list.append(iou)

    return iou_list, OA


def compute_iou(args, predl, gt_label):
    per_class_iou = np.zeros(args.num_class)

    eps = 1e-8
    for i in range(args.num_class):
        per_class_iou[i] =  (float(np.sum((predl==i)&(gt_label==i)))) / (float(np.sum((predl==i) | (gt_label==i))) + eps)

    OA = (float(np.sum(predl==gt_label))) / (float(len(gt_label)) + eps)

    return per_class_iou, OA


def acc_report(args):
    result_files = glob.glob(os.path.join(args.outdir, '*.pred_true'))
    pred_true_labels_list = []
    for i in range(len(result_files)):
        pred_true_labels_list.append(np.loadtxt(result_files[i]))
        print('%d/%d load'%(i, len(result_files)))
    pred_true_labels = np.concatenate(pred_true_labels_list, 0)

    print('computing iou...')
    iou_list, OA = compute_iou(args, np.uint8(pred_true_labels[:,0]), np.uint8(pred_true_labels[:,1]))

    print('iou: ')
    print(iou_list)
    print('mean iou: %f'% np.mean(iou_list))
    print('OA: %f' %(OA))


if __name__ == "__main__":
    test()
    interpolate()
    acc_report()
