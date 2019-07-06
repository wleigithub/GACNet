# GACNet
Graph Attention Convolution for Point Cloud Semantic Segmentation
=======

This is a Tensorflow implementation of [GACNet](http://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Graph_Attention_Convolution_for_Point_Cloud_Semantic_Segmentation_CVPR_2019_paper.html) for semantic segmentation on [S3DIS dataset](https://shapenet.cs.stanford.edu/media/indoor3d_sem_seg_hdf5_data.zip).


Installation
------

The code is tested on Ubuntu 16.04 with Python 2.7 and TF1.12.  

For data processing, [PCL](http://www.pointclouds.org/) is needed for neighbor points searching.  



Compile Customized TF Operators
-------
Most parts here are based on [PointNet++](https://github.com/charlesq34/pointnet2).  
 
The TF operators are included under tf_ops, you need to compile them first.   

Modify the path of your compiler and run    

        cd tf_ops
        sh tf_compile.sh
        
*Update nvcc and python path if necessary. The code is tested under TF1.12. If you are using earlier version it's possible that you need to remove the -D_GLIBCXX_USE_CXX11_ABI=0 flag in g++ command in order to compile correctly. 


How to use 
-----
First, you need to prepare your own dataset with the code under the folder data_processing. Slice the input scenes into blocks and down-sampling the points into a certain number, e.g., 4096.  

Here, we also calculate the geometric features in advance as it is slow to put this opteration in the traning phase. 

*[PCL](http://www.pointclouds.org/) is needed for neighbor points searching here. For a prepared dataset for S3DIS, you can download it from [here](https://drive.google.com/drive/folders/1CGY6zY0QvUG4r-DtK4axL972mhImN2bY?usp=sharing).  


After preparing the dataset, you can run    

    cd net_S3DIS
    python run.py 
    python run_test.py  
    
for training and testing on S3DIS. Other/Customized dataset can be done in a similar way.


Citation
-----
If you find our work useful in your research, please consider citing:    

@InProceedings{Wang2019_GACNet,  

author = {Wang, Lei and Huang, Yuchun and Hou, Yaolin and Zhang, Shenman and Shan, Jie},  

title = {Graph Attention Convolution for Point Cloud Semantic Segmentation},  

booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},  

month = {June},  

year = {2019}  

}  

