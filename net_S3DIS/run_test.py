import argparse
import os
import test

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log_area5', help='Log dir [default: log]')
parser.add_argument('--num_class', type=int, default=13, help='number of class')
parser.add_argument('--channel', type=int, default=12, help='number of feature channel')
parser.add_argument('--testdataset_dir', type=str, default='../Data/S3DIS/test_dataset', help='testdataset path')
parser.add_argument('--test_dir', type=str, default='/media/wl/user/3D_pcseg_Data/S3DIS/stanford_indoor3d', help='The file contains original test data')
parser.add_argument('--outdir', type=str, default='../Data/S3DIS/labels_pred_true')

args = parser.parse_args()

#graph_inf contents parameters for grapg building and coarsing
graph_inf = {'stride_list': [4, 4, 4, 2],
             'radius_list': [0.1, 0.2, 0.4, 0.8, 1.6],
             'maxsample_list': [12, 21, 21, 21, 12]
}

# number of units for each mlp layer
forward_parm = [
                [ [32,32,64], [64] ],
                [ [64,64,128], [128] ],
                [ [128,128,256], [256] ],
                [ [256,256,512], [512] ],
                [ [256,256], [256] ]
]

# for feature interpolation stage 
upsample_parm = [
                  [128, 128],
                  [128, 128],
                  [256, 256],
                  [256, 256]
]

# parameters for fully connection layer
fullconect_parm = 128

net_inf = {'forward_parm': forward_parm,
           'upsample_parm': upsample_parm,
           'fullconect_parm': fullconect_parm
}


if not os.path.exists(args.outdir): os.mkdir(args.outdir)

test.test(args, graph_inf, net_inf)
test.interpolate(args)
test.acc_report(args)
