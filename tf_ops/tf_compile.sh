#/bin/bash
TF_INC=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_include())')
TF_LIB=$(python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib())')
/usr/local/cuda-9.0/bin/nvcc cuda_ulits.cu -o cuda_ulits.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC
g++ -std=c++11 tf_sampling.cpp tf_gather.cpp tf_interpolate.cpp cuda_ulits.cu.o -o tf_op_so.so -shared -fPIC -I /usr/local/lib/python2.7/dist-packages/tensorflow/include/external/nsync/public/ -I /usr/local/lib/python2.7/dist-packages/tensorflow/include -I /usr/local/cuda-9.0/include -lcudart -L /usr/local/cuda-9.0/lib64/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0 -I $TF_INC -L $TF_LIB -ltensorflow_framework
