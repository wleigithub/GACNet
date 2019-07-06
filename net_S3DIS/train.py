

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
import provider
from model import *
import tf_util
 

def log_string(LOG_FOUT, out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def train(args, graph_inf, net_inf):

    #creat a log_dir for log record
    #if not os.path.exists(args.log_dir): os.mkdir(args.log_dir)
    LOG_FOUT = open(os.path.join(args.log_dir, 'log_train.txt'), 'a')
    LOG_FOUT.write(str(args)+'\n')

    #load data
    train_data, train_label, train_spw, test_data, test_label, test_spw = load_traindata(args.data_path, args.roomlist_file, args.test_area)
    feature_channel = train_data.shape[-1]

    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(args.gpu)):
            pointclouds_pl, labels_pl, spws_pl, is_training_pl = placeholder_inputs(feature_channel, with_spw=True)
            
            # Note the global_step=batch parameter to minimize. 
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = tf_util.get_bn_decay(batch, args.batch_size, float(args.decay_step))
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            graph_prd, coarse_map = build_graph_pyramid(pointclouds_pl[...,0:3], graph_inf)
            logits = build_network(graph_prd, coarse_map, net_inf, pointclouds_pl, args.num_class, is_training_pl, bn_decay=bn_decay)
            loss = get_loss(logits, labels_pl)
            tf.summary.scalar('loss', loss)

            pred = tf.nn.softmax(logits)
            correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(args.batch_size*args.num_point)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = tf_util.get_learning_rate(batch, args.learning_rate, args.batch_size, args.decay_step, args.decay_rate)
            tf.summary.scalar('learning_rate', learning_rate)
            if args.optimizer == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=args.momentum)
            elif args.optimizer == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(extra_update_ops):
            	train_op = optimizer.minimize(loss, global_step=batch)
            
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
        train_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'train'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(args.log_dir, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        if os.path.exists('log/checkpoint'):
            last_model = tf.train.latest_checkpoint('log/')
            saver.restore(sess, last_model)

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'spws_pl': spws_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        best_iou = -1
        for epoch in range(args.max_epoch):
            log_string(LOG_FOUT, '**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            train_one_epoch(args, sess, ops, train_writer, train_data, train_label, train_spw, LOG_FOUT)
  
            # Save the variables to disk.
            if epoch % 3 == 0:
                iou = eval_one_epoch(args, sess, ops, test_writer, test_data, test_label, test_spw, LOG_FOUT) 
                if iou>best_iou:
                    best_iou = iou
                    save_path = saver.save(sess, os.path.join(args.log_dir, "model.ckpt"), epoch)
                    log_string(LOG_FOUT, "Model saved in file: %s" % save_path)
	    #save_path = saver.save(sess, os.path.join(args.log_dir, "model.ckpt"))
    LOG_FOUT.close() #close log file


def train_one_epoch(args, sess, ops, train_writer, train_data, train_label, train_spw, LOG_FOUT):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(LOG_FOUT, '----')
    current_data, current_label, current_spw, _ = provider.shuffle_data_with_spw(train_data[:,0:args.num_point,:], train_label, train_spw) 
    current_data[:,:,0:3] = provider.rotate_point_cloud_along_z(current_data[:,:,0:3]) # rotate along z-axis

    file_size = current_data.shape[0]
    num_batches = file_size // args.batch_size
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    for batch_idx in range(num_batches):
        if batch_idx % 100 == 0:
            print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * args.batch_size
        end_idx = (batch_idx+1) * args.batch_size
        
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['spws_pl']: current_spw[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                                feed_dict=feed_dict)

        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])
        total_correct += correct
        total_seen += (args.batch_size*args.num_point)
        loss_sum += loss_val
        print('loss: %f' % loss_val)
        print('accuracy: %f' % (correct/float(args.batch_size*args.num_point)))
    
    log_string(LOG_FOUT, 'mean loss: %f' % (loss_sum / float(num_batches)))
    log_string(LOG_FOUT, 'accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(args, sess, ops, test_writer, test_data, test_label, test_spw, LOG_FOUT):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(args.num_class)]
    total_correct_class = [0 for _ in range(args.num_class)]
    
    log_string(LOG_FOUT, '----')
    current_data = test_data[:,0:args.num_point,:]
    current_label = np.squeeze(test_label)
    
    file_size = current_data.shape[0]
    num_batches = file_size // args.batch_size

    gt_classes = [0 for _ in range(args.num_class)]
    positive_classes = [0 for _ in range(args.num_class)]
    true_positive_classes = [0 for _ in range(args.num_class)]
    
    for batch_idx in range(num_batches):
        if (batch_idx%20==0):
            print('eval: %d / %d' %(batch_idx, num_batches))
        start_idx = batch_idx * args.batch_size
        end_idx = (batch_idx+1) * args.batch_size

        batch_label = current_label[start_idx:end_idx,:]
        batch_spw = test_spw[start_idx:end_idx,:]
        feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: batch_label,
                     ops['spws_pl']: batch_spw,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum((pred_val == batch_label) & (batch_spw>0))
        total_correct += correct
        total_seen += np.sum(batch_spw>0)
        loss_sum += (loss_val*args.batch_size)
        for i in range(start_idx, end_idx):
            for j in range(args.num_point):
                if test_spw[i,j]==0: continue
                l = current_label[i, j]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx, j] == l)
          
        gt_classes, positive_classes, true_positive_classes = provider.utils_iou(\
                    gt_classes, positive_classes, true_positive_classes, pred_val, batch_label, batch_spw)
    iou_list = []
    for i in range(args.num_class):
        iou = true_positive_classes[i]/float(gt_classes[i]+positive_classes[i]-true_positive_classes[i]) 
        iou_list.append(iou)
          
    log_string(LOG_FOUT, 'eval mean loss: %f' % (loss_sum / float(total_seen/args.num_point)))
    log_string(LOG_FOUT, 'eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string(LOG_FOUT, 'eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
    log_string(LOG_FOUT, 'eval avg iou: %f' % (np.mean(iou_list)))
     
    return np.mean(iou_list)
        

def load_traindata(data_path, roomlist_file, test_area):
    ALL_FILES = glob.glob(data_path + '/*.h5')
    room_filelist = [line.rstrip() for line in open(roomlist_file)]

    # Load ALL data
    data_batch_list = []
    label_batch_list = []
    spw_batch_list = []
    for h5_filename in ALL_FILES:
        data_batch, label_batch, spw_batch = provider.loadDataFile_with_spw(h5_filename)
        data_batch_list.append(data_batch)
        label_batch_list.append(label_batch)
        spw_batch_list.append(spw_batch)
    data_batches = np.concatenate(data_batch_list, 0)
    label_batches = np.concatenate(label_batch_list, 0)
    spw_batches = np.concatenate(spw_batch_list, 0)

    test_area = 'Area_'+str(test_area)
    train_idxs = []
    test_idxs = []
    for i,room_name in enumerate(room_filelist):
        if test_area in room_name:
            test_idxs.append(i)
        else:
            train_idxs.append(i)

    train_data = data_batches[train_idxs,...]
    train_label = label_batches[train_idxs]
    train_spw = spw_batches[train_idxs]

    test_data = data_batches[test_idxs,...]
    test_label = label_batches[test_idxs]
    test_spw = spw_batches[test_idxs]
    print(train_data.shape, train_label.shape)
    print(test_data.shape, test_label.shape)

    return train_data, train_label, train_spw, test_data, test_label, test_spw
