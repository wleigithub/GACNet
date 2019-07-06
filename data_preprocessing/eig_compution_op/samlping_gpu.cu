#include <unistd.h>
#include <stdio.h>
/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
// sample m points from n points
// input (n,3)
// output (m,3)
__global__ void farthestpointsamplingKernel(int n,int m, const int random_init, float *temp, const float *dataset, int *idxs){

  const int BlockSize=512;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  
  int old_id = random_init;
  idxs[0] = old_id;

  for (int j=threadIdx.x;j<n;j+=blockDim.x){
    temp[j]=1e38;
  }

  for(int i=1;i<m;i++){
      int besti=0;
      float best=-1;
      float x0 = dataset[old_id*3+0];
      float y0 = dataset[old_id*3+1];
      float z0 = dataset[old_id*3+2];
      
      for(int j=threadIdx.x;j<n;j+=blockDim.x){
          float td=temp[j];

          float x1 = dataset[j*3+0];
          float y1 = dataset[j*3+1];
          float z1 = dataset[j*3+2];

          float dist = (x1-x0)*(x1-x0)+(y1-y0)*(y1-y0)+(z1-z0)*(z1-z0);

          float d2=min(dist,td);
          if (d2!=td)
            temp[j]=d2;
          if (d2>best){
            best=d2;
            besti=j;
          }
      }

      dists[threadIdx.x]=best;
      dists_i[threadIdx.x]=besti;
      for (int u=0;(1<<u)<blockDim.x;u++){
        __syncthreads();
        if (threadIdx.x<(blockDim.x>>(u+1))){
          int i1=(threadIdx.x*2)<<u;
          int i2=(threadIdx.x*2+1)<<u;
          if (dists[i1]<dists[i2]){
            dists[i1]=dists[i2];
            dists_i[i1]=dists_i[i2];
          }
        }
      }
      __syncthreads();
      old_id=dists_i[0];
      if (threadIdx.x==0)
        idxs[i]=old_id;
    }

}
extern "C" void farthestpointsampling(int n, int m, const int random_init, float *temp_host, const float *dataset_host, int *idxs_host){
    float *temp, *dataset;
    int *idxs;
    cudaError_t error;

    cudaMalloc((void**)&dataset, sizeof(float)* n*3);  
    cudaMalloc((void**)&temp, sizeof(float)* n);
    cudaMalloc((void**)&idxs, sizeof(int)* m); 

    cudaMemcpy(dataset, dataset_host, sizeof(float)* n*3, cudaMemcpyHostToDevice);
    cudaMemcpy(temp, temp_host, sizeof(float)* n, cudaMemcpyHostToDevice);
    cudaMemcpy(idxs, idxs_host, sizeof(int)* m, cudaMemcpyHostToDevice);

    farthestpointsamplingKernel<<<1, 512>>>(n, m, random_init, temp, dataset, idxs);

    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
        printf("code: %d, reason: %s\n",error,cudaGetErrorString(error));
    }
    
    cudaMemcpy(idxs_host, idxs, sizeof(int)*m, cudaMemcpyDeviceToHost);
    cudaFree(temp);
    cudaFree(dataset);
    cudaFree(idxs);
}


__global__ void fps_multiblocks_Kernel(int num_sample, int num_block, const int *num_pts, const int *start_id, const int *sort_id, const int *seed, float *temp, const float *dataset, int *idxs){

    const int BlockSize=512;
    __shared__ float dists[BlockSize];
    __shared__ int dists_i[BlockSize];
    

    for (int i=blockIdx.x;i<num_block;i+=gridDim.x){

        if(num_pts[i]==0){
            printf("num_pts cannot be zero!");
            return;
        }
        if(num_pts[i]<num_sample ){
            for (int j=threadIdx.x;j<num_sample;j+=blockDim.x){
                int tid = j;             
                if(j>=num_pts[i]){
                    curandState_t state;
                    curand_init(0, 0, 0,&state);
                    tid = curand(&state) % num_pts[i];
                }
                idxs[i*num_sample+j]=sort_id[ start_id[i]+tid ];
            }

        }
        else if(num_pts[i]==num_sample ){
            for (int j=threadIdx.x;j<num_sample;j+=blockDim.x)
                idxs[i*num_sample+j]=sort_id[ start_id[i]+j ];
        }
        else{
            int old=seed[i];
            //printf("old: %d\t", old);
            idxs[i*num_sample]=sort_id[old+start_id[i]];
            for (int j=threadIdx.x;j<num_pts[i];j+=blockDim.x){
                temp[start_id[i] + j]=1e38;
            }
    
            for (int j=1;j<num_sample;j++){
                int besti=0;
                float best=-1;
                float x1=dataset[start_id[i]*3 + old*3+0];
                float y1=dataset[start_id[i]*3 + old*3+1];
                float z1=dataset[start_id[i]*3 + old*3+2];
                for (int k=threadIdx.x;k<num_pts[i];k+=blockDim.x){
                    float td=temp[start_id[i] + k];
                    float x2,y2,z2;
                    x2=dataset[start_id[i]*3 + k*3+0];
                    y2=dataset[start_id[i]*3 + k*3+1];
                    z2=dataset[start_id[i]*3 + k*3+2];
                    
                    float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
                    float d2=min(d,td);
                    if (d2!=td)
                        temp[start_id[i] + k]=d2;
                    if (d2>best){
                        best=d2;
                        besti=k;
                    }
                }
                dists[threadIdx.x]=best;
                dists_i[threadIdx.x]=besti;
                for (int u=0;(1<<u)<blockDim.x;u++){
                  __syncthreads();
                  if (threadIdx.x<(blockDim.x>>(u+1))){
                    int i1=(threadIdx.x*2)<<u;
                    int i2=(threadIdx.x*2+1)<<u;
                    if (dists[i1]<dists[i2]){
                      dists[i1]=dists[i2];
                      dists_i[i1]=dists_i[i2];
                    }
                  }
                }
                __syncthreads();
                old=dists_i[0];
                //if(j<3)
                //    printf("old2: %d\t", old);
                if (threadIdx.x==0)
                  idxs[i*num_sample+j]=sort_id[old+start_id[i]];

            }
        }
    }
}


/* this GPU kernel function is used to initialize the random states */
__global__ void init(unsigned int seed, curandState_t* states) {
    /* we have to initialize the state */
    curand_init(seed, /* the seed can be the same for each core, here we pass the time in from the CPU */
                blockIdx.x, /* the sequence number should be different for each core (unless you want all
                               cores to get the same sequence of numbers for some reason - use thread id! */
                0, /* the offset is how much extra we advance in the sequence for each call, can be 0 */
                &states[blockIdx.x]);
}
  
/* this GPU kernel takes an array of states, and an array of ints, and puts a random int into each */
__global__ void randoms(curandState_t* states, int* numbers, int *upbound) {
    /* curand works like rand - except that it takes a state as a parameter */
    numbers[blockIdx.x] = curand(&states[blockIdx.x]) % upbound[blockIdx.x];
}


__global__ void print(int *data, int num){
    for(int i=0;i<num;i++)
        printf("init_seed: %d\t", data[i]);
}

extern "C" void fps_multiblocks(int totalnum_pts, int num_sample, int num_block, const int *num_pts_host, const int *start_id_host, const int *sort_id_host, const float *dataset_host, int *idxs_host){

    float *temp, *dataset;
    int *num_pts, *start_id, *idxs, *seed, *sort_id;
    cudaError_t error;
    curandState_t* states;

    cudaMalloc((void**)&dataset, sizeof(float)* totalnum_pts*3);  
    cudaMalloc((void**)&temp, sizeof(float)* totalnum_pts);
    cudaMalloc((void**)&sort_id, sizeof(int)* totalnum_pts);
    cudaMalloc((void**)&idxs, sizeof(int)* num_sample*num_block); 
    cudaMalloc((void**)&num_pts, sizeof(int)*num_block);
    cudaMalloc((void**)&start_id, sizeof(int)*num_block);
    cudaMalloc((void**)&seed, sizeof(int)*num_block);
    cudaMalloc((void**) &states, num_block * sizeof(curandState_t));


    cudaMemcpy(dataset, dataset_host, sizeof(float)* totalnum_pts*3, cudaMemcpyHostToDevice);
    cudaMemcpy(sort_id, sort_id_host, sizeof(int)* totalnum_pts, cudaMemcpyHostToDevice);
    cudaMemcpy(num_pts, num_pts_host, sizeof(int)* num_block, cudaMemcpyHostToDevice);
    cudaMemcpy(start_id, start_id_host, sizeof(int)* num_block, cudaMemcpyHostToDevice);
    cudaMemcpy(idxs, idxs_host, sizeof(int)* num_sample*num_block, cudaMemcpyHostToDevice);

    init<<<num_block, 1>>>(time(0), states);
    randoms<<<num_block, 1>>>(states, seed, num_pts);

    
    fps_multiblocks_Kernel<<<64, 512>>>(num_sample, num_block, num_pts, start_id, sort_id,seed, temp, dataset, idxs);

    //int batchsize=16;
    //int num_batch=num_block/batchsize+1;
    //for (int i=0;i<num_batch;i++){
    //    fps_multiblocks_Kernel<<<batchsize, 256>>>(i, num_sample, num_block, num_pts, start_id, sort_id,seed, temp, dataset, idxs);
    //}
    error = cudaDeviceSynchronize();
    if(error != cudaSuccess){
        printf("code: %d, reason: %s\n",error,cudaGetErrorString(error));
    }
    /*
    printf("num_block: %d\n", num_block);
    print<<<1,1>>>(start_id,num_block);
    printf("num_pts: \n");
    print<<<1,1>>>(num_pts,num_block);
    */

    cudaMemcpy(idxs_host, idxs, sizeof(int)*num_sample*num_block, cudaMemcpyDeviceToHost);

    cudaFree(states);
    cudaFree(temp);
    cudaFree(dataset);
    cudaFree(idxs);
    cudaFree(start_id);
    cudaFree(num_pts);
    cudaFree(seed);
    cudaFree(sort_id);
}




