from eig_feature import FPS
import numpy as np
import pdb
from plyfile import PlyData, PlyElement
def read_xyzrgbIL_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    pc = plydata['vertex'].data
    pc_array = np.array([[x, y, z, red, green, blue, label] for x,y,z, red, green, blue, label in pc])
    return pc_array


if __name__=='__main__':
    datafilename = '/media/wl/myhome/wl/wanglei/Code/transequnet/Data/stanford_indoor/stanford_indoor3d/Area_1_copyRoom_1.ply' 
    org_data = read_xyzrgbIL_ply(datafilename)
    
    xyz = org_data[:,0:3]
    #pdb.set_trace()
    idxs = FPS(xyz, 1000)
    new_xyz = xyz[idxs,...]
    pdb.set_trace()
    np.savetxt('/home/wl/1.txt', new_xyz)
