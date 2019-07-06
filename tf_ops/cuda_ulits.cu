#include <unistd.h>
#include <stdio.h>

/* we need these includes for CUDA's random number stuff */
#include <curand.h>
#include <curand_kernel.h>
__global__ void farthestpointsamplingKernel(int b,int n,int m, const int *init, const float * __restrict__ dataset,float * __restrict__ temp,int * __restrict__ idxs){
  if (m<=0)
    return;
  const int BlockSize=512;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  const int BufferSize=3072;
  __shared__ float buf[BufferSize*3];

  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    int old=init[i];
    if (threadIdx.x==0)
      idxs[i*m+0]=old;
    for (int j=threadIdx.x;j<n;j+=blockDim.x){
      temp[blockIdx.x*n+j]=1e38;
    }
    for (int j=threadIdx.x;j<min(BufferSize,n)*3;j+=blockDim.x){
      buf[j]=dataset[i*n*3+j];
    }
    __syncthreads();
    for (int j=1;j<m;j++){
      int besti=0;
      float best=-1;
      float x1=dataset[i*n*3+old*3+0];
      float y1=dataset[i*n*3+old*3+1];
      float z1=dataset[i*n*3+old*3+2];
      for (int k=threadIdx.x;k<n;k+=blockDim.x){
        float td=temp[blockIdx.x*n+k];
        float x2,y2,z2;
        if (k<BufferSize){
          x2=buf[k*3+0];
          y2=buf[k*3+1];
          z2=buf[k*3+2];
        }else{
          x2=dataset[i*n*3+k*3+0];
          y2=dataset[i*n*3+k*3+1];
          z2=dataset[i*n*3+k*3+2];
        }
        float d=(x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1);
        float d2=min(d,td);
        if (d2!=td)
          temp[blockIdx.x*n+k]=d2;
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
      if (threadIdx.x==0)
        idxs[i*m+j]=old;
    }
  }
}

// input: radius (1), num (1), data_xyz (b,n,3), search_xyz (b,m,3)
// output: idx (b,m,num), dist (b,m, num)
__global__ void query_ball_knn_gpu(int b, int n, int m, float radius, int num, const int *shuffled_ids, const float *data_xyz, const float *search_xyz, int *idx, float *dist) {
    int batch_index = blockIdx.x;
    data_xyz += n*3*batch_index;
    search_xyz += m*3*batch_index;
    shuffled_ids += n*batch_index;
    idx += m*num*batch_index;
    dist += m*num*batch_index; // counting how many unique points selected in local region

    for (int j=threadIdx.x;j<m;j+=blockDim.x){
        float search_x=search_xyz[j*3+0];
        float search_y=search_xyz[j*3+1];
        float search_z=search_xyz[j*3+2];

        int sort_id = 0; float bigest= 0;
        bool is_full=false; 
        
        for (int l=0;l<num;++l){
            dist[j*num + l]=99999.0;
            idx[j*num + l]=0;
        } 

        for (int k=0;k<n;++k) {
            //find the bigest and its id
            bigest = dist[j*num];
            sort_id = 0;
            for (int l=1;l<num;++l){
                if(dist[j*num + l]>bigest){
                    bigest = dist[j*num + l];
                    sort_id = l;
                } 
            }
            if(bigest<radius){
                is_full=true;
                break;
            } 

            int kk= shuffled_ids[k];
            float data_x=data_xyz[kk*3+0];
            float data_y=data_xyz[kk*3+1];
            float data_z=data_xyz[kk*3+2];
            float d=max(sqrtf((data_x-search_x)*(data_x-search_x)+(data_y-search_y)*(data_y-search_y)+(data_z-search_z)*(data_z-search_z)),1e-20f);
            
            //replace the bigest one
            if(bigest>d){
                dist[j*num + sort_id] = d;
                idx[j*num + sort_id] = kk;
            }
        }

        //if the nearghbors are less than k_num
        if (is_full || bigest<90000.0) continue;
        for(int k=0;k<num;++k){
            if(dist[j*num + k]>90000.0){
                    dist[j*num + k] = dist[j*num];
                    idx[j*num + k] = idx[j*num];	        
            }
        }
    }
}

__global__ void query_ball_gpu(int b, int n, int m, float radius, int nsample, const int *shuffled_ids, const float *data_xyz, const float *search_xyz, int *idx, int *pts_cnt) {
    int batch_index = blockIdx.x;
    data_xyz += n*3*batch_index;
    search_xyz += m*3*batch_index;
    idx += m*nsample*batch_index;
    pts_cnt += m*batch_index; // counting how many unique points selected in local region

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        int cnt = 0;
        float search_x=search_xyz[j*3+0];
        float search_y=search_xyz[j*3+1];
        float search_z=search_xyz[j*3+2];

        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball

            int kk= shuffled_ids[k];
            float data_x=data_xyz[kk*3+0];
            float data_y=data_xyz[kk*3+1];
            float data_z=data_xyz[kk*3+2];
            float d=max(sqrtf((data_x-search_x)*(data_x-search_x)+(data_y-search_y)*(data_y-search_y)+(data_z-search_z)*(data_z-search_z)),1e-20f);

            if (d<radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        idx[j*nsample+l] = kk;
                }
                idx[j*nsample+cnt] = kk;
                cnt+=1;
            }
        }
        pts_cnt[j] = cnt;
    }
}

//k nearst search with cuda
__global__ void knn_cuda(int b, int n, int m, int num, const float *data_xyz, const float *search_xyz, int *idx, float *dist)
{
    int batch_index = blockIdx.x;
    data_xyz += n*3*batch_index;
    search_xyz += m*3*batch_index;
    idx += m*num*batch_index;
    dist += m*num*batch_index;

    for (int j=threadIdx.x;j<m;j+=blockDim.x){
        float search_x=search_xyz[j*3+0];
        float search_y=search_xyz[j*3+1];
        float search_z=search_xyz[j*3+2];

        int sort_id = 0; float bigest= 0;
	    int tmp_id=0; float tmp_dist = 99999.0;

        for (int l=0;l<num;++l){
            dist[j*num + l]=99999.0;
            idx[j*num + l]=0;
        } 

        for (int k=0;k<n;++k) {
            float data_x=data_xyz[k*3+0];
            float data_y=data_xyz[k*3+1];
            float data_z=data_xyz[k*3+2];
            float d=max(sqrtf((data_x-search_x)*(data_x-search_x)+(data_y-search_y)*(data_y-search_y)+(data_z-search_z)*(data_z-search_z)),1e-20f);

            if(d<1e-10f) {
                tmp_dist = 0.0;
                tmp_id = k;
            continue;
            }

            //find the bigest and its id
            bigest = dist[j*num];
            sort_id = 0;
            for (int l=1;l<num;++l){
                if(dist[j*num + l]>bigest){
                    bigest = dist[j*num + l];
                    sort_id = l;
                } 
            }

            //replace the bigest one
            if(bigest>d){
                dist[j*num + sort_id] = d;
                idx[j*num + sort_id] = k;
            }
        } 

        //put itself into the results
        if(tmp_dist<1e-10f){
            //find the bigest and its id
            bigest = dist[j*num];
            sort_id = 0;
            for (int l=1;l<num;++l){
                if(dist[j*num + l]>dist[j*num + l-1]){
                    bigest = dist[j*num + l];
                    sort_id = l;
                } 
            }
            dist[j*num + sort_id] = 0.0;
            idx[j*num + sort_id] = tmp_id;
        }
        //if the nearghbors are less than k_num
        for(int k=0;k<num;++k){
            if(dist[j*num + k]>90000.0){
                    dist[j*num + k] = dist[j*num];
                    idx[j*num + k] = idx[j*num];	        
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
__global__ void randoms(curandState_t* states, int* numbers, int upbound) {
  /* curand works like rand - except that it takes a state as a parameter */
  numbers[blockIdx.x] = curand(&states[blockIdx.x]) % upbound;
}

void randomLauncher(int b, int *gpu_nums, int upbound){

  /* CUDA's random number library uses curandState_t to keep track of the seed value
     we will store a random state for every thread  */
  curandState_t* states;

  /* allocate space on the GPU for the random states */
  cudaMalloc((void**) &states, b * sizeof(curandState_t));

  /* invoke the GPU to initialize all of the random states */
  init<<<b, 1>>>(time(0), states);

  /* allocate an array of unsigned ints on the CPU and GPU */
  //int cpu_nums[b];
  //unsigned int* gpu_nums;
  //cudaMalloc((void**) &gpu_nums, b * sizeof(unsigned int));

  /* invoke the kernel to get some random numbers */
  randoms<<<b, 1>>>(states, gpu_nums, upbound);

  /* copy the random numbers back */
  //cudaMemcpy(cpu_nums, gpu_nums, b * sizeof(int), cudaMemcpyDeviceToHost);

  /* print them out */
  //for (int i = 0; i < b; i++) {
  //  printf("%u\n", cpu_nums[i]);
  //}
  //printf("upbound: %u\n", upbound);

  /* free the memory we allocated for the states and numbers */
  cudaFree(states);
  //cudaFree(gpu_nums);
}

void queryBallLauncher(int b, int n, int m, float radius, int k_num, const int *shuffled_ids, const float *data_xyz, const float *search_xyz, int *idx, int *pts_cnt) {
    query_ball_gpu<<<b,256>>>(b,n,m,radius,k_num,shuffled_ids, data_xyz, search_xyz,idx,pts_cnt);
    //cudaDeviceSynchronize();
}

//require 32*n working space
void farthestpointsamplingLauncher(int b,int n,int m,const int *init, const float * inp,float * temp,int * out){
    farthestpointsamplingKernel<<<32,512>>>(b,n,m,init, inp,temp,out);
}

void queryBallKnnLauncher(int b, int n, int m, float radius, int k_num, const int *shuffled_ids, const float *data_xyz, const float *search_xyz, int *idx, float *dist) {
    query_ball_knn_gpu<<<b,256>>>(b,n,m,radius,k_num,shuffled_ids, data_xyz, search_xyz,idx,dist);
    //check_gpu<<<b,256>>>(b,  n,  m,  nsample,idx);    
    //cudaDeviceSynchronize();
}

void knnLauncher(int b, int n, int m, int k_num, const float *data_xyz, const float *search_xyz, int *idx, float *dist){
    knn_cuda<<<b, 512>>>(b, n, m, k_num, data_xyz, search_xyz, idx, dist);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// input: inp (b,n,c), idx (b,m)
// output: out (b,m,c)
__global__ void gatherpointKernel(int b,int n,int m,int channel, const float * __restrict__ inp,const int * __restrict__ idx,float * __restrict__ out){
    int batch_index = blockIdx.x;
    inp += n*channel*batch_index;
    idx += m*batch_index;
    out += m*channel*batch_index;
    for(int i = threadIdx.x;i<m;i+=blockDim.x){
        int a = idx[i];
        for(int j=0;j<channel;++j) out[i*channel + j]=inp[a*channel + j];
    }
}

__global__ void gatheridKernel(int b,int n,int m,int channel, const int * __restrict__ inp,const int * __restrict__ idx,int * __restrict__ out){
    int batch_index = blockIdx.x;
    inp += n*channel*batch_index;
    idx += m*batch_index;
    out += m*channel*batch_index;
    for(int i = threadIdx.x;i<m;i+=blockDim.x){
        int a = idx[i];
        for(int j=0;j<channel;++j) out[i*channel + j]=inp[a*channel + j];
    }
}

// input: out_g(b,m,c), idx (b,m)
// output: inp_g(b,n,c)
__global__ void scatteraddpointKernel(int b,int n,int m,int channel, const float * __restrict__ out_g,const int * __restrict__ idx,float * __restrict__ inp_g){
    int batch_index = blockIdx.x;
    inp_g += n*channel*batch_index;
    idx += m*batch_index;
    out_g += m*channel*batch_index;
    for(int i = threadIdx.x;i<m;i+=blockDim.x){
        int a = idx[i];
        for(int j=0;j<channel;j++) 
            inp_g[a*channel + j] += out_g[i*channel + j];
    }
}

__global__ void scatteraddidKernel(int b,int n,int m,int channel, const int * __restrict__ out_g,const int * __restrict__ idx,int * __restrict__ inp_g){
    int batch_index = blockIdx.x;
    inp_g += n*channel*batch_index;
    idx += m*batch_index;
    out_g += m*channel*batch_index;
    for(int i = threadIdx.x;i<m;i+=blockDim.x){
        int a = idx[i];
        for(int j=0;j<channel;++j) 
            atomicAdd(&inp_g[a*channel + j],out_g[i*channel + j]);
    }
}

// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    int batch_index = blockIdx.x;
    points += n*c*batch_index;
    idx += m*nsample*batch_index;
    out += m*nsample*c*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<m;j+=stride) {
        for (int k=0;k<nsample;++k) {
            int ii = idx[j*nsample+k];
            for (int l=0;l<c;++l) {
                out[j*nsample*c+k*c+l] = points[ii*c+l];
            }
        }
    }
}

// input: grad_out (b,m,nsample,c), idx (b,m,nsample), 
// output: grad_points (b,n,c)
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
    
    int batch_index = blockIdx.x;
    grad_points += n*c*batch_index;
    idx += m*nsample*batch_index;
    grad_out += m*c*nsample*batch_index;
    int nsc= c*nsample;

    for(int i=0;i<m;i++){
        for(int j=0;j<nsample;j++){
            int a = idx[i*nsample+j];
            for(int k=threadIdx.x;k<c;k+=blockDim.x){
                grad_points[a*c+k] += grad_out[i*nsc + j*c + k];
            }        
        }
    }   
}

// input: eigvectors(eigvs): (b,n,3,3), idx: (b,n)
// output: out (b,n,3)
__global__ void gather_eigvector_gpu(int b, int n, const float *eigvs, const int *idx, float *out) {
    int batch_index = blockIdx.x;
    eigvs += n*9*batch_index;
    idx += n*batch_index;
    out += n*3*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<n;j+=stride) {
	int ii = idx[j];
	out[j*3] = eigvs[j*9+ii];
	out[j*3+1] = eigvs[j*9+ii+3];
	out[j*3+2] = eigvs[j*9+ii+6];
    }
}

// input: grad_out: (b,n,3), idx: (b,n)
// output: grad_eigvs (b,n,3, 3)
__global__ void gather_eigvector_grad_gpu(int b, int n, const float *grad_out, const int *idx, float *grad_eigvs) {
    int batch_index = blockIdx.x;
    grad_eigvs += n*9*batch_index;
    idx += n*batch_index;
    grad_out += n*3*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;
    
    for (int j=index;j<n;j+=stride) {
	int ii = idx[j];
	atomicAdd(&grad_eigvs[j*9+ii], grad_out[j*3]);
	atomicAdd(&grad_eigvs[j*9+ii+3], grad_out[j*3+1]);
	atomicAdd(&grad_eigvs[j*9+ii+6], grad_out[j*3+2]);
    }
}

void gatherpointLauncher(int b,int n,int m,int channel, const float * inp,const int * idx,float * out){
  gatherpointKernel<<<b,256>>>(b,n,m,channel, inp,idx,out);
}
void gatheridLauncher(int b,int n,int m,int channel, const int * inp,const int * idx,int * out){
  gatheridKernel<<<b,256>>>(b,n,m,channel,inp,idx,out);
}
void scatteraddpointLauncher(int b,int n,int m,int channel, const float * out_g,const int * idx,float * inp_g){
  scatteraddpointKernel<<<b,256>>>(b,n,m,channel,out_g,idx,inp_g);
}
void scatteraddidLauncher(int b,int n,int m,int channel, const int * out_g,const int * idx,int * inp_g){
  scatteraddidKernel<<<b,256>>>(b,n,m,channel,out_g,idx,inp_g);
}

void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out){
    //group_point_gpu<<<dim3(b,nsample,1),256>>>(b,n,c,m,nsample,points,idx,out);
    group_point_gpu<<<b,256>>>(b,n,c,m,nsample,points,idx,out);
    //cudaDeviceSynchronize();
}
void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points){
    group_point_grad_gpu<<<b,c>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //cudaDeviceSynchronize();
}

void gatherEigvectorLauncher(int b, int n, const float *eigvs, const int *idx, float *out){
    gather_eigvector_gpu<<<b,256>>>(b,n,eigvs,idx,out);
    //cudaDeviceSynchronize();
}
void gatherEigvectorGradLauncher(int b, int n, const float *grad_out, const int *idx, float *grad_eigvs){
    gather_eigvector_grad_gpu<<<b,256>>>(b,n,grad_out,idx,grad_eigvs);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// input: points (b,m,c), idx (b,n,3), weight (b,n,3)
// output: out (b,n,c)
__global__ void threeinterpolate_kernel(int b, int m, int c, int n, const float * __restrict__ points, const int * __restrict__ idx, const float * __restrict__ weight, float * out) {
    int batch_index = blockIdx.x;
    points += m*c*batch_index;
    idx += n*3*batch_index;
    weight += n*3*batch_index;
    out += n*c*batch_index;

    float w1,w2,w3;
    int i1,i2,i3;
    for (int i=threadIdx.x;i<n;i+=blockDim.x){
        w1=weight[i*3];
        w2=weight[i*3 + 1];
        w3=weight[i*3 + 2]; 
        i1=idx[i*3];
        i2=idx[i*3 + 1];
        i3=idx[i*3 + 2]; 
        for (int j=0;j<c;++j) {
            out[i*c+j] = points[i1*c+j]*w1 + points[i2*c+j]*w2 + points[i3*c+j]*w3;
        }
    }
}

// input: grad_out (b,n,c), idx (b,n,3), weight (b,n,3)
// output: grad_points (b,m,c)
__global__ void threeinterpolate_grad_kernel(int b, int n, int c, int m, const float * __restrict__ grad_out, const int * __restrict__ idx, const float * __restrict__ weight, float * grad_points) {
    int batch_index = blockIdx.x;
    grad_points += m*c*batch_index;
    idx += n*3*batch_index;
    weight += n*3*batch_index;
    grad_out += n*c*batch_index;

    float w1,w2,w3;
    int i1,i2,i3;
    for (int i=0;i<n;i++){
        w1=weight[i*3];
        w2=weight[i*3 + 1];
        w3=weight[i*3 + 2]; 
        i1=idx[i*3];
        i2=idx[i*3 + 1];
        i3=idx[i*3 + 2]; 
        for (int j=threadIdx.x;j<c;j+=blockDim.x) {
            grad_points[i1*c+j] += grad_out[i*c+j]*w1;
            grad_points[i2*c+j] += grad_out[i*c+j]*w2;
            grad_points[i3*c+j] += grad_out[i*c+j]*w3;
        }
    }
}

void threeinterpolateLauncher(int b, int m, int c, int n, const float *points, const int *idx, const float *weight, float *out) {
    threeinterpolate_kernel<<<b, 256>>>(b, m, c, n, points, idx, weight, out);
}
void threeinterpolategradLauncher(int b, int n, int c, int m, const float *grad_out, const int *idx, const float *weight, float *grad_points) {
    threeinterpolate_grad_kernel<<<b, c>>>(b, n, c, m, grad_out, idx, weight, grad_points);
}
