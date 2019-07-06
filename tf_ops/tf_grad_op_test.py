import tensorflow as tf
import numpy as np
import random
from tf_ops import knn_search, three_interpolate, group_point, gather_point, FPS_downsample, query_ball_point, query_circle_point, shuffle_ids
import pdb

class TfOpsTest(tf.test.TestCase):
  def test(self):
    pass

  def test_grad(self):
    with self.test_session():
      with tf.device('/gpu:0'):
        points = tf.constant(np.random.random((1,68,16)).astype('float32'))
        print(points)
        xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
        xyz2 = tf.constant(np.random.random((1,68,3)).astype('float32'))

        print ('---------------------group------------------')
        idxs, dist = knn_search(xyz1, xyz2, 5)
        #idxs, dist = query_ball_point(0.05, 5, xyz1, xyz2)
        idxs, dist = query_circle_point(0.0, 1.1, 5, xyz1, xyz2)
        with tf.device('/cpu:0'):
          shuffled_ids = shuffle_ids(xyz1)
        with tf.Session() as sess:  
          idx_out, dist_, shuffled_ids_ = sess.run([idxs, dist, shuffled_ids]) 
        idx= tf.constant(idx_out)

        print(idx)
        grouped_points = group_point(xyz1, idxs)
        print(grouped_points)
        err = tf.test.compute_gradient_error(xyz1, (1,128,3), grouped_points, (1,68,5,3))
        print(err)
        self.assertLess(err, 1e-4)  

        print ('---------------------interpolate------------------')
        idx, dist = knn_search(xyz1, xyz2, 3)
        print(idx)
        weight = tf.ones_like(dist)/3.0
        interpolated_points = three_interpolate(points, idx, weight)
        print(interpolated_points)
        err = tf.test.compute_gradient_error(points, (1,68,16), interpolated_points, (1,128,16))
        print(err)
        self.assertLess(err, 1e-4) 

        print ('---------------------gather------------------')
        idxs = FPS_downsample(xyz1, 100)
        with tf.Session() as sess:  
          idx_out = sess.run(idxs)  
        idx= tf.constant(idx_out)
        print(idx)
        gathered_points = gather_point(xyz1, idx)
        print(gathered_points)
        err = tf.test.compute_gradient_error(xyz1, (1,128,3), gathered_points, (1,100,3))
        print(err)
        self.assertLess(err, 1e-4)    


if __name__=='__main__':
  '''
  xyz1 = tf.constant(np.random.random((1,128,3)).astype('float32'))
  idx = FPS_downsample(xyz1, 100)
  with tf.device('/gpu:0'):
    with tf.Session() as sess:  
      idx_out = sess.run(idx)  
  pdb.set_trace()
  '''
  tf.test.main() 
