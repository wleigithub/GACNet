import os
import numpy as np
import sys
import indoor3d_util
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(BASE_DIR))
sys.path.append(BASE_DIR)

org_datadir = '/media/wl/user/3D_pcseg_Data/S3DIS' #modify this direction according to your own data path

# Constants     
indoor3d_data_dir = os.path.join(org_datadir, 'stanford_indoor3d')
NUM_POINT = 4096
H5_BATCH_SIZE = 100000
data_dim = [NUM_POINT, 12]
label_dim = [NUM_POINT]
data_dtype = 'float32'
label_dtype = 'uint8'
spw_dtype = 'float32'

# Set paths
filelist = os.path.join(BASE_DIR, 'meta/all_data_label.txt')
data_label_files = [os.path.join(indoor3d_data_dir, line.split('.')[0]+'.ply') for line in open(filelist)]
output_dir = os.path.join(ROOT_DIR, 'Data/S3DIS/train_dataset')
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
output_filename_prefix = os.path.join(output_dir, 'ply_data_all')
output_room_filelist = os.path.join(output_dir, 'room_filelist.txt')
fout_room = open(output_room_filelist, 'w')

# --------------------------------------
# ----- BATCH WRITE TO HDF5 -----
# --------------------------------------
batch_data_dim = [H5_BATCH_SIZE] + data_dim
batch_label_dim = [H5_BATCH_SIZE] + label_dim
batch_spw_dim = [H5_BATCH_SIZE] + label_dim
h5_batch_data = np.zeros(batch_data_dim, dtype = np.float32)
h5_batch_label = np.zeros(batch_label_dim, dtype = np.uint8)
h5_batch_spw = np.zeros(batch_spw_dim, dtype = np.float32)
buffer_size = 0  # state: record how many samples are currently in buffer
h5_index = 0 # state: the next h5 file to save

def insert_batch(data, label, spw, last_batch=False):
    global h5_batch_data, h5_batch_label
    global buffer_size, h5_index
    data_size = data.shape[0]
    # If there is enough space, just insert
    if buffer_size + data_size <= h5_batch_data.shape[0]:
        h5_batch_data[buffer_size:buffer_size+data_size, ...] = data
        h5_batch_label[buffer_size:buffer_size+data_size] = label
        h5_batch_spw[buffer_size:buffer_size+data_size] = spw
        buffer_size += data_size
    else: # not enough space
        capacity = h5_batch_data.shape[0] - buffer_size
        assert(capacity>=0)
        if capacity > 0:
           h5_batch_data[buffer_size:buffer_size+capacity, ...] = data[0:capacity, ...] 
           h5_batch_label[buffer_size:buffer_size+capacity, ...] = label[0:capacity, ...] 
           h5_batch_spw[buffer_size:buffer_size+capacity, ...] = spw[0:capacity, ...]
        # Save batch data and label to h5 file, reset buffer_size
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        indoor3d_util.save_h5(h5_filename, h5_batch_data, h5_batch_label, h5_batch_spw, data_dtype, label_dtype, spw_dtype) 
        print('Stored {0} with size {1}'.format(h5_filename, h5_batch_data.shape[0]))
        h5_index += 1
        buffer_size = 0
        # recursive call
        insert_batch(data[capacity:, ...], label[capacity:, ...], spw[capacity:, ...], last_batch)
    if last_batch and buffer_size > 0:
        h5_filename =  output_filename_prefix + '_' + str(h5_index) + '.h5'
        indoor3d_util.save_h5(h5_filename, h5_batch_data[0:buffer_size, ...], h5_batch_label[0:buffer_size, ...], h5_batch_spw[0:buffer_size, ...], data_dtype, label_dtype, spw_dtype)
        print('Stored {0} with size {1}'.format(h5_filename, buffer_size))
        h5_index += 1
        buffer_size = 0
    return


sample_cnt = 0
room_filelist = []
for i, data_label_filename in enumerate(data_label_files):
    print(data_label_filename)
    data, label, spw = indoor3d_util.scene2blocks_zeromean(data_label_filename, NUM_POINT, block_size=1.2, inner_size=1.0, stride=0.6, radius=0.1)
    print('{0}, {1}'.format(data.shape, label.shape))
    for _ in range(data.shape[0]):
        fout_room.write(os.path.basename(data_label_filename)[0:-4]+'\n')

    sample_cnt += data.shape[0]
    insert_batch(data, label, spw, i == len(data_label_files)-1)

fout_room.close()
print("Total samples: {0}".format(sample_cnt))
