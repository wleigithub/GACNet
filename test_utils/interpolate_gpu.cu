#include "cuda_runtime.h"  
#include "device_launch_parameters.h"
#include "stdio.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// input: features (n,c), idx (n,3), weight (n,3)
// output: probs (n,c)
__global__ void interpolate_kernel(int n, int c, int k_num, float *features,  float *weight,  int *idx, float *probs) {

    for(int i= blockIdx.x; i<n; i+=gridDim.x){
        for(int j=threadIdx.x; j<c;j+=blockDim.x){
            float tmp_prob = 0;
            for(int k=0; k<k_num;k++){
                int idd = idx[i*k_num + k];
                float wgt = weight[i*k_num + k];
                tmp_prob += features[idd*c+j]*wgt;      
            }
            probs[i*c+j] += tmp_prob;
            //printf("%f\t", tmp_prob);
        }
    }
}

extern "C" void interpolateLauncher(int n_host, int m_host, int c_host, float *features_host, int *idx_host, float *weight_host, float *probs_host){
    //int *n_dev, *c_dev;
    float *weight, *features, *probs;
    int *idx;
    cudaError_t error;

    cudaMalloc((void**)&weight, sizeof(float)* n_host*3);  
    cudaMalloc((void**)&idx, sizeof(int)* n_host*3); 
    cudaMalloc((void**)&features, sizeof(float)* m_host*c_host);  
    cudaMalloc((void**)&probs, sizeof(float)* n_host*c_host);  

    cudaMemcpy(weight, weight_host, sizeof(float)* n_host*3, cudaMemcpyHostToDevice);
    cudaMemcpy(idx, idx_host, sizeof(int)* n_host*3, cudaMemcpyHostToDevice);
    cudaMemcpy(features, features_host, sizeof(float)* m_host*c_host, cudaMemcpyHostToDevice);
    cudaMemcpy(probs, probs_host, sizeof(float)*n_host*c_host, cudaMemcpyHostToDevice);

    dim3 grid(32768, 1, 1), block(c_host, 3, 1);
    interpolate_kernel<<<grid, block>>>(n_host, c_host, 3, features, weight, idx, probs);
    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
        printf("code: %d, reason: %s\n",error,cudaGetErrorString(error));
    }


    cudaMemcpy(probs_host, probs, sizeof(float)*n_host*c_host, cudaMemcpyDeviceToHost);

    cudaFree(weight);
    cudaFree(features);
    cudaFree(probs);
    cudaFree(idx);
}


extern "C" void filterLauncher(int n_host, int c_host, int k_num, int iter_num, float *features_host, int *idx_host, float *weight_host, float *probs_host){
    //int *n_dev, *c_dev;
    float *weight, *features, *probs;
    int *idx;
    cudaError_t error;

    cudaMalloc((void**)&weight, sizeof(float)* n_host*k_num);  
    cudaMalloc((void**)&idx, sizeof(int)* n_host*k_num); 
    cudaMalloc((void**)&features, sizeof(float)* n_host*c_host);  
    cudaMalloc((void**)&probs, sizeof(float)* n_host*c_host);  

    cudaMemcpy(weight, weight_host, sizeof(float)* n_host*k_num, cudaMemcpyHostToDevice);
    cudaMemcpy(idx, idx_host, sizeof(int)* n_host*k_num, cudaMemcpyHostToDevice);
    cudaMemcpy(features, features_host, sizeof(float)* n_host*c_host, cudaMemcpyHostToDevice);
    cudaMemcpy(probs, probs_host, sizeof(float)*n_host*c_host, cudaMemcpyHostToDevice);

    dim3 grid(32768, 1, 1), block(c_host, 1, 1);
    for(int i=0;i<iter_num;i++){
        interpolate_kernel<<<grid, block>>>(n_host, c_host, k_num, features, weight, idx, probs);
        error = cudaDeviceSynchronize();
        if(error != cudaSuccess){
            printf("code: %d, reason: %s\n",error,cudaGetErrorString(error));
        }
    }

    cudaMemcpy(probs_host, probs, sizeof(float)*n_host*c_host, cudaMemcpyDeviceToHost);

    cudaFree(weight);
    cudaFree(features);
    cudaFree(probs);
    cudaFree(idx);
}
