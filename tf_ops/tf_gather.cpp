#include <cstdio>
#include <ctime>
#include <cstring> // memset
#include <cstdlib> // rand, RAND_MAX
#include <cmath> // sqrtf
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>
using namespace tensorflow;

REGISTER_OP("GatherPoint")
  .Input("inp: float32")
  .Input("idx: int32")
  .Output("out: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims0; // batch_size * ndataset * 3
    c->WithRank(c->input(0), 3, &dims0);
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoints
    c->WithRank(c->input(1), 2, &dims1);
    // batch_size * npoints * channel
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims0, 0), c->Dim(dims1, 1), c->Dim(dims0, 2)});
    c->set_output(0, output);
    return Status::OK();
  });
REGISTER_OP("GatherPointGrad")
  .Input("inp: float32")
  .Input("idx: int32")
  .Input("out_g: float32")
  .Output("inp_g: float32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    c->set_output(0, c->input(0));
    return Status::OK();
  });
REGISTER_OP("GatherId")
  .Input("inp: int32")
  .Input("idx: int32")
  .Output("out: int32")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    ::tensorflow::shape_inference::ShapeHandle dims0; // batch_size * ndataset * 3
    c->WithRank(c->input(0), 3, &dims0);
    ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoints
    c->WithRank(c->input(1), 2, &dims1);
    // batch_size * npoints * channel
    ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims0, 0), c->Dim(dims1, 1), c->Dim(dims0, 2)});
    c->set_output(0, output);
    return Status::OK();
  });

REGISTER_OP("GroupPoint")
    .Input("points: float32")
    .Input("idx: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims0; // batch_size * ndataset * channels
        c->WithRank(c->input(0), 3, &dims0);
        ::tensorflow::shape_inference::ShapeHandle dims1; // batch_size * npoints * nsample
        c->WithRank(c->input(1), 3, &dims1);
        // batch_size * npoints * nsample * channels
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims1, 0), c->Dim(dims1, 1), c->Dim(dims1, 2), c->Dim(dims0, 2)});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("GroupPointGrad")
    .Input("points: float32")
    .Input("idx: int32")
    .Input("grad_out: float32")
    .Output("grad_points: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

REGISTER_OP("GatherEigvector")
    .Input("eigvs: float32")
    .Input("idx: int32")
    .Output("out: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        ::tensorflow::shape_inference::ShapeHandle dims0; // batch_size * ndataset * 3 *3
        c->WithRank(c->input(0), 4, &dims0);
        ::tensorflow::shape_inference::ShapeHandle output = c->MakeShape({c->Dim(dims0, 0), c->Dim(dims0, 1), 3});
        c->set_output(0, output);
        return Status::OK();
    });
REGISTER_OP("GatherEigvectorGrad")
    .Input("eigvs: float32")
    .Input("idx: int32")
    .Input("grad_out: float32")
    .Output("grad_eigvs: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        c->set_output(0, c->input(0));
        return Status::OK();
    });

void gatherpointLauncher(int b,int n,int m,int channel, const float * inp,const int * idx,float * out);
class GatherPointGpuOp: public OpKernel{
  public:
    explicit GatherPointGpuOp(OpKernelConstruction * context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3,errors::InvalidArgument("GatherPoint expects (batch_size,num_points,channel) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      int channel=inp_tensor.shape().dim_size(2);
      const Tensor& idx_tensor=context->input(1);
      OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b,errors::InvalidArgument("GatherPoint expects (batch_size,num_result) idx shape"));
      int m=idx_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      auto idx_flat=idx_tensor.flat<int>();
      const int * idx=&(idx_flat(0));
      Tensor * out_tensor=NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,channel},&out_tensor));
      auto out_flat=out_tensor->flat<float>();
      float * out=&(out_flat(0));
      gatherpointLauncher(b,n,m,channel,inp,idx,out);
    }
};
REGISTER_KERNEL_BUILDER(Name("GatherPoint").Device(DEVICE_GPU),GatherPointGpuOp);

void scatteraddpointLauncher(int b,int n,int m,int channel, const float * out_g,const int * idx,float * inp_g);
class GatherPointGradGpuOp: public OpKernel{
  public:
    explicit GatherPointGradGpuOp(OpKernelConstruction * context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,num_points,channel) inp"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      int channel=inp_tensor.shape().dim_size(2);
      const Tensor& idx_tensor=context->input(1);
      OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,num_result) idx shape"));
      int m=idx_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<float>();
      const float * inp=&(inp_flat(0));
      auto idx_flat=idx_tensor.flat<int>();
      const int * idx=&(idx_flat(0));
      const Tensor& out_g_tensor=context->input(2);
      OP_REQUIRES(context,out_g_tensor.dims()==3 && out_g_tensor.shape().dim_size(0)==b && out_g_tensor.shape().dim_size(1)==m,errors::InvalidArgument("GatherPointGradGpuOp expects (batch_size,num_result,channel) out_g shape"));
      auto out_g_flat=out_g_tensor.flat<float>();
      const float * out_g=&(out_g_flat(0));
      Tensor * inp_g_tensor=nullptr;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,n,channel},&inp_g_tensor));
      auto inp_g_flat=inp_g_tensor->flat<float>();
      float * inp_g=&(inp_g_flat(0));
      cudaMemset(inp_g,0,sizeof(float)*b*n*channel);
      scatteraddpointLauncher(b,n,m,channel,out_g,idx,inp_g);
    }
};
REGISTER_KERNEL_BUILDER(Name("GatherPointGrad").Device(DEVICE_GPU),GatherPointGradGpuOp);

void gatheridLauncher(int b,int n,int m,int channel, const int * inp,const int * idx,int * out);
class GatherIdGpuOp: public OpKernel{
  public:
    explicit GatherIdGpuOp(OpKernelConstruction * context):OpKernel(context){}
    void Compute(OpKernelContext * context)override{
      const Tensor& inp_tensor=context->input(0);
      OP_REQUIRES(context,inp_tensor.dims()==3,errors::InvalidArgument("GatherId expects (batch_size,num_points,num_sampling1) inp shape"));
      int b=inp_tensor.shape().dim_size(0);
      int n=inp_tensor.shape().dim_size(1);
      int channel=inp_tensor.shape().dim_size(2);
      const Tensor& idx_tensor=context->input(1);
      OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b,errors::InvalidArgument("GatherId expects (batch_size,num_sampling2) idx shape"));
      int m=idx_tensor.shape().dim_size(1);
      auto inp_flat=inp_tensor.flat<int>();
      const int * inp=&(inp_flat(0));
      auto idx_flat=idx_tensor.flat<int>();
      const int * idx=&(idx_flat(0));
      Tensor * out_tensor=NULL;
      OP_REQUIRES_OK(context,context->allocate_output(0,TensorShape{b,m,channel},&out_tensor));
      auto out_flat=out_tensor->flat<int>();
      int * out=&(out_flat(0));
      gatheridLauncher(b,n,m,channel,inp,idx,out);
    }
};
REGISTER_KERNEL_BUILDER(Name("GatherId").Device(DEVICE_GPU),GatherIdGpuOp);

void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out);
class GroupPointGpuOp: public OpKernel{
    public:
        explicit GroupPointGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPoint expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPoint expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,m,nsample,c}, &out_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            groupPointLauncher(b,n,c,m,nsample,points,idx,out);
        }
};
REGISTER_KERNEL_BUILDER(Name("GroupPoint").Device(DEVICE_GPU),GroupPointGpuOp);

void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points);
class GroupPointGradGpuOp: public OpKernel{
    public:
        explicit GroupPointGradGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& points_tensor=context->input(0);
            OP_REQUIRES(context, points_tensor.dims()==3, errors::InvalidArgument("GroupPointGrad expects (batch_size, num_points, channel) points shape"));
            int b = points_tensor.shape().dim_size(0);
            int n = points_tensor.shape().dim_size(1);
            int c = points_tensor.shape().dim_size(2);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==3 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GroupPointGrad expects (batch_size, npoints, nsample) idx shape"));
            int m = idx_tensor.shape().dim_size(1);
            int nsample = idx_tensor.shape().dim_size(2);

            const Tensor& grad_out_tensor=context->input(2);
            OP_REQUIRES(context,grad_out_tensor.dims()==4 && grad_out_tensor.shape().dim_size(0)==b && grad_out_tensor.shape().dim_size(1)==m && grad_out_tensor.shape().dim_size(2)==nsample && grad_out_tensor.shape().dim_size(3)==c, errors::InvalidArgument("GroupPointGrad expects (batch_size, npoints, nsample, channel) grad_out shape"));

            Tensor * grad_points_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,c}, &grad_points_tensor));

            auto points_flat = points_tensor.flat<float>();
            const float *points = &(points_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto grad_out_flat = grad_out_tensor.flat<float>();
            const float *grad_out = &(grad_out_flat(0));
            auto grad_points_flat = grad_points_tensor->flat<float>();
            float *grad_points = &(grad_points_flat(0));
            cudaMemset(grad_points, 0, sizeof(float)*b*n*c);
            groupPointGradLauncher(b,n,c,m,nsample,grad_out,idx,grad_points);
        }
};
REGISTER_KERNEL_BUILDER(Name("GroupPointGrad").Device(DEVICE_GPU),GroupPointGradGpuOp);


void gatherEigvectorLauncher(int b, int n, const float *eigvs, const int *idx, float *out);
class GatherEigvectorGpuOp: public OpKernel{
    public:
        explicit GatherEigvectorGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& eigvs_tensor=context->input(0);
            OP_REQUIRES(context, eigvs_tensor.dims()==4, errors::InvalidArgument("GatherEigvector expects (batch_size, num_points, 3, 3) points shape"));
            int b = eigvs_tensor.shape().dim_size(0);
            int n = eigvs_tensor.shape().dim_size(1);
            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GatherEigvector expects (batch_size, npoints) idx shape"));

            Tensor * out_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,3}, &out_tensor));

            auto eigvs_flat = eigvs_tensor.flat<float>();
            const float *eigvs = &(eigvs_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto out_flat = out_tensor->flat<float>();
            float *out = &(out_flat(0));
            gatherEigvectorLauncher(b,n,eigvs,idx,out);
        }
};
REGISTER_KERNEL_BUILDER(Name("GatherEigvector").Device(DEVICE_GPU),GatherEigvectorGpuOp);


void gatherEigvectorLauncher(int b, int n, const float *grad_out, const int *idx, float *grad_eigvs);
class GatherEigvectorGradGpuOp: public OpKernel{
    public:
        explicit GatherEigvectorGradGpuOp(OpKernelConstruction * context):OpKernel(context){}

        void Compute(OpKernelContext * context) override {
            const Tensor& eigvs_tensor=context->input(0);
            OP_REQUIRES(context, eigvs_tensor.dims()==4, errors::InvalidArgument("GatherEigvectorGrad expects (batch_size, num_points, 3, 3) points shape"));
            int b = eigvs_tensor.shape().dim_size(0);
            int n = eigvs_tensor.shape().dim_size(1);

            const Tensor& idx_tensor=context->input(1);
            OP_REQUIRES(context,idx_tensor.dims()==2 && idx_tensor.shape().dim_size(0)==b, errors::InvalidArgument("GatherEigvectorGrad expects (batch_size, num_points) idx shape"));

            const Tensor& grad_out_tensor=context->input(2);
            OP_REQUIRES(context,grad_out_tensor.dims()==3 && grad_out_tensor.shape().dim_size(0)==b && grad_out_tensor.shape().dim_size(1)==n, errors::InvalidArgument("GatherEigvectorGrad expects (batch_size, npoints, nsample, channel) grad_out shape"));

            Tensor * grad_eigvs_tensor = nullptr;
            OP_REQUIRES_OK(context, context->allocate_output(0,TensorShape{b,n,3,3}, &grad_eigvs_tensor));

            auto eigvs_flat = eigvs_tensor.flat<float>();
            const float *eigvs = &(eigvs_flat(0));
            auto idx_flat = idx_tensor.flat<int>();
            const int *idx = &(idx_flat(0));
            auto grad_out_flat = grad_out_tensor.flat<float>();
            const float *grad_out = &(grad_out_flat(0));
            auto grad_eigvs_flat = grad_eigvs_tensor->flat<float>();
            float *grad_eigvs = &(grad_eigvs_flat(0));
            cudaMemset(grad_eigvs, 0, sizeof(float)*b*n*3*3);
            gatherEigvectorLauncher(b,n,grad_out,idx,grad_eigvs);
        }
};
REGISTER_KERNEL_BUILDER(Name("GatherEigvectorGrad").Device(DEVICE_GPU),GatherEigvectorGradGpuOp);