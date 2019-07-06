#include "cuda_runtime.h"  
#include "device_launch_parameters.h"
#include "stdio.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
__global__ void isinblock_kernel(float *xy,  float x_low, float y_low, float size, bool *is_block, int N) {
 
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < N){ 
        float x = xy[idx*2];
        float y = xy[idx*2+1];
        float x_up = x_low+size;
        float y_up = y_low+size;

        if (x>x_low && x<x_up && y>y_low && y<y_up)
            is_block[idx] = true;

        idx += blockDim.x*gridDim.x;
    }

}

extern "C" void is_in_block(float *xy_host, bool *is_block_host, float &x_low, float &y_low, float &size, int &N){
    float *xy;
    bool *is_block;
    cudaError_t error;

    cudaMalloc((void**)&xy, sizeof(float)* N*2);  
    cudaMalloc((void**)&is_block, sizeof(bool)* N); 

    cudaMemcpy(xy, xy_host, sizeof(float)* N*2, cudaMemcpyHostToDevice);
    cudaMemcpy(is_block, is_block_host, sizeof(bool)* N, cudaMemcpyHostToDevice);

    isinblock_kernel<<<32786, 256>>>(xy, x_low, y_low, size, is_block, N);

    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
        printf("code: %d, reason: %s\n",error,cudaGetErrorString(error));
    }

    cudaMemcpy(is_block_host, is_block, sizeof(float)*N, cudaMemcpyDeviceToHost);
    cudaFree(xy);
    cudaFree(is_block);
}

