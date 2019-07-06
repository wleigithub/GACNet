
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(three_interpolate)

# packages
find_package(CUDA REQUIRED)

# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50)

#set(CUDA_NVCC_FLAGS -gencode arch=compute_20,code=sm_20;-G;-g)
#set(CUDA_NVCC_FLAGS -gencode arch=compute_52,code=sm_52;-G;-g)

FIND_PACKAGE( OpenMP REQUIRED)  
if(OPENMP_FOUND)  
  message("OPENMP FOUND")  
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  

#set(flannlibs "/usr/lib/x86_64-linux-gnu/libflann.so")
include_directories("/usr/local/cuda-9.0/include" "usr/include/flann")
link_directories("/usr/local/cuda-9.0/lib64/")

set(flannlibs "/usr/lib/x86_64-linux-gnu/libflann.so")
set(src three_interpolate.cpp interpolate_gpu.cu)

CUDA_ADD_LIBRARY(three_interpolate SHARED ${src})
target_link_libraries (three_interpolate ${flannlibs})
