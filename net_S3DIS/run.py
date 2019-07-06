import argparse
import os
import train

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_class', type=int, default=13, help='number of class')
parser.add_argument('--num_point', type=int, default=4096, help='Point number [default: 4096]')
parser.add_argument('--max_epoch', type=int, default=50, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=15, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')
parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
parser.add_argument('--data_path', type=str, default='../Data/S3DIS/train_dataset', help='data path')
parser.add_argument('--roomlist_file', type=str, default='../Data/S3DIS/train_dataset/room_filelist.txt', help='The file which recorded the room name of each train data ')
args = parser.parse_args()


#graph_inf contents parameters for grapg building and coarsing
graph_inf = {'stride_list': [4, 4, 4, 2], #can be seen as the downsampling rate
             'radius_list': [0.1, 0.2, 0.4, 0.8, 1.6], # radius for neighbor points searching 
             'maxsample_list': [12, 21, 21, 21, 12] #number of neighbor points for each layer
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


if not os.path.exists(args.log_dir): os.mkdir(args.log_dir)
os.system('cp run.py %s' % (args.log_dir)) # back up parameter inf
os.system('cp train.py %s' % (args.log_dir)) # back up parameter inf
os.system('cp ../utils/pcnet_util.py %s' % (args.log_dir)) # back up parameter inf
os.system('cp ../utils/model.py %s' % (args.log_dir)) # back up parameter inf
train.train(args, graph_inf, net_inf)
