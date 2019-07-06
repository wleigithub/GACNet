#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
#include <iostream>
#include <algorithm> 
#include <stdlib.h> 
#include <time.h>  

using namespace tensorflow;



REGISTER_OP("RandSeeds")
  .Input("inp: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
    c->WithRank(c->input(0), 3, &dims1);
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0)});
    c->set_output(0, output);
    return Status::OK();
  });


REGISTER_OP("ShuffleIds")
  .Input("inp: float32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
    c->WithRank(c->input(0), 3, &dims1);
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1)});
    c->set_output(0, output);
    return Status::OK();
  });


Status GetFPSNum(
    shape_inference::InferenceContext* c,
    shape_inference::DimensionHandle input_size,
    int64 stride,
    shape_inference::DimensionHandle* output_size) {
  if (stride <= 0) {
    return errors::InvalidArgument("Stride must be > 0, but got ", stride);
  }
  TF_RETURN_IF_ERROR(c->Add(input_size, stride - 1, output_size));
  TF_RETURN_IF_ERROR(c->Divide(*output_size, stride, false /* evenly_divisible */, output_size));
  return Status::OK();
}

REGISTER_OP("FarthestPointSample")
  .Attr("stride: int")
  .Input("inp: float32")
  .Input("initid: int32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoint * 3
    c->WithRank(c->input(0), 3, &dims1);

    ::tensorflow::shape_inference::DimensionHandle org_pcnum = c->Dim(dims1, 1);
    int stride;
    TF_RETURN_IF_ERROR(c->GetAttr("stride", &stride));
    ::tensorflow::shape_inference::DimensionHandle npoint;
    GetFPSNum(c, org_pcnum, stride, &npoint);

    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), npoint});
    c->set_output(0, output);
    return Status::OK();
  });


REGISTER_OP("QueryBallKnn")
    .Attr("radius: float")
    .Attr("k_num: int")
    .Input("data_xyz: float32")
    .Input("search_xyz: float32")
    .Input("shuffled_ids: int32")
    .Output("idx: int32")
    .Output("dist: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int k_num;
        TF_RETURN_IF_ERROR(c->GetAttr("k_num", &k_num));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), k_num});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), k_num});
        c->set_output(1, output2);
        return Status::OK();
    });


REGISTER_OP("QueryBall")
    .Attr("radius: float")
    .Attr("k_num: int")
    .Input("data_xyz: float32")
    .Input("search_xyz: float32")
    .Input("shuffled_ids: int32")
    .Output("idx: int32")       
    .Output("pts_cnt: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int k_num;
        TF_RETURN_IF_ERROR(c->GetAttr("k_num", &k_num));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), k_num});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1)});
        c->set_output(1, output2);
        return Status::OK();
    });


REGISTER_OP("KnnSearch")
    .Attr("k_num: int")
    .Input("data_xyz: float32")
    .Input("search_xyz: float32")
    .Output("idx: int32")
    .Output("dist: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims2; // batch_size * npoint * 3
        c->WithRank(c->input(1), 3, &dims2);
        int k_num;      
        TF_RETURN_IF_ERROR(c->GetAttr("k_num", &k_num));
        ::tensorflow::shape_inference::ShapeHandle output1 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), k_num});
        c->set_output(0, output1);
        ::tensorflow::shape_inference::ShapeHandle output2 = c->MakeShape({c->Dim(dims2, 0), c->Dim(dims2, 1), k_num});
        c->set_output(1, output2);
        return Status::OK();
    });

void randomLauncher(int b, int *gpu_nums, int upbound);
class RandSeedGpuOp: public OpKernel{
  public:
    explicit RandSeedGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext * context)override{

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,errors::InvalidArgument("RandSeedGpuOp expects (batch_size,num_points,3) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      randomLauncher(b, out, n);
    }
};
REGISTER_KERNEL_BUILDER(Name("RandSeeds").Device(DEVICE_GPU),RandSeedGpuOp);


void shuffleidcpu(int b, int n, int *shuffled_ids){
    //initialization
    for(int i=0;i<b*n;i++)
        shuffled_ids[i] = i%n;
    //shuffle ids
    for(int i=0;i<b;i++)
        std::random_shuffle(&shuffled_ids[i*n],&shuffled_ids[i*n+n]);
}
class ShuffleIdOp: public OpKernel{
  public:
    explicit ShuffleIdOp(OpKernelConstruction* context) : OpKernel(context) {}

    void Compute(OpKernelContext * context)override{

      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,errors::InvalidArgument("ShuffleIdOp expects (batch_size,num_points,3) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b, n},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      shuffleidcpu(b, n, out);
    }
};
REGISTER_KERNEL_BUILDER(Name("ShuffleIds").Device(DEVICE_CPU),ShuffleIdOp);

void farthestpointsamplingLauncher(int b,int n,int m,const int *init, const float * inp,float * temp,int * out);
class FarthestPointSampleGpuOp: public OpKernel{
  public:
    explicit FarthestPointSampleGpuOp(OpKernelConstruction* context):OpKernel(context) {
                    OP_REQUIRES_OK(context, context->GetAttr("stride", &stride_));
                    OP_REQUIRES(context, stride_ > 0, errors::InvalidArgument("FarthestPointSample expects positive stride"));
                }
    void Compute(OpKernelContext * context)override{
      
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3 && inp_tensor.shape().dim_size(2)==3,errors::InvalidArgument("FarthestPointSample expects (batch_size,num_points,3) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      int m = (n+stride_-1)/stride_; //point number to sample
      Tensor * out_tensor;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      Tensor temp_tensor;
      OP_REQUIRES_OK(context,context->allocate_temp(DataTypeToEnum<float>::value,TensorShape{32,n},&temp_tensor));
      auto temp_flat=temp_tensor.flat<float>();
      float * temp=&(temp_flat(0));

      const Tensor& initid_tensor=context->input(1);
      //Tensor& initid_tensor=context->input(1);
      //OP_REQUIRES(context,inp_tensor.dims()==1 ,errors::InvalidArgument("FarthestPointSample expects (batch_size,num_points,3) inp shape"));
      auto initid_flat=initid_tensor.flat<int>();
      const int * init=&(initid_flat(0));
      farthestpointsamplingLauncher(b,n,m,init, inp,temp,out);

    }
    private:
        int stride_;
};
REGISTER_KERNEL_BUILDER(Name("FarthestPointSample").Device(DEVICE_GPU),FarthestPointSampleGpuOp);


void queryBallKnnLauncher(int b, int n, int m, float radius, int k_num, const int *shuffled_ids, const float *data_xyz, const float *search_xyz, int *idx, float *dist);
class QueryBallKnnGpuOp : public OpKernel {
    public:
        explicit QueryBallKnnGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryBallKnn expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("k_num", &k_num_));
            OP_REQUIRES(context, k_num_ > 0, errors::InvalidArgument("QueryBallKnn expects positive k_num"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& data_xyz_tensor = context->input(0);
            OP_REQUIRES(context, data_xyz_tensor.dims()==3 && data_xyz_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallKnn expects (batch_size, ndataset, 3) data_xyz shape."));
            int b = data_xyz_tensor.shape().dim_size(0);
            int n = data_xyz_tensor.shape().dim_size(1);

            const Tensor& search_xyz_tensor = context->input(1);
            OP_REQUIRES(context, search_xyz_tensor.dims()==3 && search_xyz_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBallKnn expects (batch_size, npoint, 3) search_xyz shape."));
            int m = search_xyz_tensor.shape().dim_size(1);

            const Tensor& shuffled_ids_tensor = context->input(2);
            OP_REQUIRES(context, shuffled_ids_tensor.dims()==2, errors::InvalidArgument("QueryBallKnn expects (batch_size, ndataset) shuffled_ids shape."));

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,k_num_}, &idx_tensor));
            Tensor *dist_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m,k_num_}, &dist_tensor));

            auto shuffled_ids_flat = shuffled_ids_tensor.flat<int>();
            const int *shuffled_ids = &(shuffled_ids_flat(0));
            auto data_xyz_flat = data_xyz_tensor.flat<float>();
            const float *data_xyz = &(data_xyz_flat(0));
            auto search_xyz_flat = search_xyz_tensor.flat<float>();
            const float *search_xyz = &(search_xyz_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto dist_flat = dist_tensor->flat<float>();
            float *dist = &(dist_flat(0));

            queryBallKnnLauncher(b,n,m,radius_,k_num_,shuffled_ids,data_xyz,search_xyz,idx,dist);
        }
    private:
        float radius_;
        int k_num_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBallKnn").Device(DEVICE_GPU), QueryBallKnnGpuOp);


void queryBallLauncher(int b, int n, int m, float radius, int k_num, const int *shuffled_ids, const float *data_xyz, const float *search_xyz, int *idx, int *pts_cnt);
class QueryBallGpuOp : public OpKernel {
    public:
        explicit QueryBallGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("radius", &radius_));
            OP_REQUIRES(context, radius_ > 0, errors::InvalidArgument("QueryBall expects positive radius"));

            OP_REQUIRES_OK(context, context->GetAttr("k_num", &k_num_));
            OP_REQUIRES(context, k_num_ > 0, errors::InvalidArgument("QueryBall expects positive k_num"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& data_xyz_tensor = context->input(0);
            OP_REQUIRES(context, data_xyz_tensor.dims()==3 && data_xyz_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBall expects (batch_size, ndataset, 3) data_xyz shape."));
            int b = data_xyz_tensor.shape().dim_size(0);
            int n = data_xyz_tensor.shape().dim_size(1);

            const Tensor& search_xyz_tensor = context->input(1);
            OP_REQUIRES(context, search_xyz_tensor.dims()==3 && search_xyz_tensor.shape().dim_size(2)==3, errors::InvalidArgument("QueryBall expects (batch_size, npoint, 3) search_xyz shape."));
            int m = search_xyz_tensor.shape().dim_size(1);

            const Tensor& shuffled_ids_tensor = context->input(2);
            OP_REQUIRES(context, shuffled_ids_tensor.dims()==2, errors::InvalidArgument("QueryBall expects (batch_size, ndataset) shuffled_ids shape."));

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,k_num_}, &idx_tensor));
            Tensor *pts_cnt_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m}, &pts_cnt_tensor));

            auto shuffled_ids_flat = shuffled_ids_tensor.flat<int>();
            const int *shuffled_ids = &(shuffled_ids_flat(0));
            auto data_xyz_flat = data_xyz_tensor.flat<float>();
            const float *data_xyz = &(data_xyz_flat(0));
            auto search_xyz_flat = search_xyz_tensor.flat<float>();
            const float *search_xyz = &(search_xyz_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto pts_cnt_flat = pts_cnt_tensor->flat<int>();
            int *pts_cnt = &(pts_cnt_flat(0));

            queryBallLauncher(b,n,m,radius_,k_num_,shuffled_ids,data_xyz,search_xyz,idx,pts_cnt);
        }
    private:
        float radius_;
        int k_num_;
};
REGISTER_KERNEL_BUILDER(Name("QueryBall").Device(DEVICE_GPU), QueryBallGpuOp);

void knnLauncher(int b, int n, int m, int k_num, const float *data_xyz, const float *search_xyz, int *idx, float *dist);
class KnnSearchGpuOp : public OpKernel {
    public:
        explicit KnnSearchGpuOp(OpKernelConstruction* context) : OpKernel(context) {
            OP_REQUIRES_OK(context, context->GetAttr("k_num", &k_num_));
            OP_REQUIRES(context, k_num_ > 0, errors::InvalidArgument("KnnSearch expects positive k_num"));
        }

        void Compute(OpKernelContext* context) override {
            const Tensor& data_xyz_tensor = context->input(0);
            OP_REQUIRES(context, data_xyz_tensor.dims()==3 && data_xyz_tensor.shape().dim_size(2)==3, errors::InvalidArgument("KnnSearch expects (batch_size, ndataset, 3) xyz1 shape."));
            int b = data_xyz_tensor.shape().dim_size(0);
            int n = data_xyz_tensor.shape().dim_size(1);

            const Tensor& search_xyz_tensor = context->input(1);
            OP_REQUIRES(context, search_xyz_tensor.dims()==3 && search_xyz_tensor.shape().dim_size(2)==3, errors::InvalidArgument("KnnSearch expects (batch_size, npoint, 3) xyz2 shape."));
            int m = search_xyz_tensor.shape().dim_size(1);

            Tensor *idx_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,m,k_num_}, &idx_tensor));
            Tensor *dist_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape{b,m,k_num_}, &dist_tensor));

            auto data_xyz_flat = data_xyz_tensor.flat<float>();
            const float *data_xyz = &(data_xyz_flat(0));
            auto search_xyz_flat = search_xyz_tensor.flat<float>();
            const float *search_xyz = &(search_xyz_flat(0));
            auto idx_flat = idx_tensor->flat<int>();
            int *idx = &(idx_flat(0));
            auto dist_flat = dist_tensor->flat<float>();
            float *dist = &(dist_flat(0));

            knnLauncher(b, n, m, k_num_, data_xyz, search_xyz, idx, dist);
        }
    
    private:
        int k_num_;
};
REGISTER_KERNEL_BUILDER(Name("KnnSearch").Device(DEVICE_GPU), KnnSearchGpuOp);

