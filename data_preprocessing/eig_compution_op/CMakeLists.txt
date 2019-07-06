
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(eig_com)

find_package(PCL 1.8 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS} )
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

MESSAGE(STATUS "This is PCL_INCLUDE_DIRS dir " ${PCL_INCLUDE_DIRS})
MESSAGE(STATUS "This is PCL_LIBRARY_DIRS dir " ${PCL_LIBRARY_DIRS})
MESSAGE(STATUS "This is PCL_DEFINITIONS dir " ${PCL_DEFINITIONS})
MESSAGE(STATUS "This is PCL_LIBRARIES dir " ${PCL_LIBRARIES})

FIND_PACKAGE( OpenMP REQUIRED)  
if(OPENMP_FOUND)  
  message("OPENMP FOUND")  
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")  
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")  
endif()  

# packages
find_package(CUDA REQUIRED)
# nvcc flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50)
#set(flannlibs "/usr/lib/x86_64-linux-gnu/libflann.so")

set(src eig_com.cpp scene2blocks.cpp samlping_gpu.cu)

CUDA_ADD_LIBRARY(eig_com SHARED ${src})
target_link_libraries (eig_com ${PCL_LIBRARIES})
