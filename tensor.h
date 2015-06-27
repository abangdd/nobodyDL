#ifndef TENSOR_H_
#define TENSOR_H_

#include <memory.h>

#include "util.h"
#include "expr.h"
#include "xpu.h"

class Shape {
public:
  explicit Shape ();
  explicit Shape (const int a, const int b, const int c, const int d);
  explicit Shape (const int a, const int b, const int c, const int d, const int e);
  bool operator == (const Shape &s) const;
  bool operator != (const Shape &s) const;
  void set_dims ();
  void set_nums (const int n);
  void set_chls (const int c);
  void set_cols (const int c);
  int get_dimsX (const int d) const;
  int get_sizeX (const int d) const;
  int get_strdX (const int d) const;
  void re_shape (const int a, const int b, const int c, const int d);
  void print ();
  int rows, cols, chls, nums;
  int size, dims;
};

class Patch {
public:
  explicit Patch () { }
  explicit Patch (int a, int b, int c) : ksize(a), pad(b), stride(c)  { }
  Shape get_pack_size (const Shape &in);  // 会改变patch
  int ksize, pad, stride;
  int h_col, w_col;
};

class Pool {
public:
  explicit Pool () { }
  explicit Pool (int a, int b, int c) : ksize(a), pad(b), stride(c)  { }
  Shape get_pool_size (const Shape &in);  // 会改变pool
  int ksize, pad, stride;
  int h_pool, w_pool;
};



enum rand_t
{ GAUSSIAN	= 1,
  UNIFORM	= 2
};

void rand_check (const int status);

template <typename XPU>
class Random {
public:
  explicit Random (const int did);
  ~Random();
  void set_seed (int seed);
  void gaussian (float *data, int size, const float mu, const float sigma) const;
  void uniform  (float *data, int size, const float  a, const float b)     const;
private:
  int did_;
#ifdef __CUDACC__
  curandGenerator_t randGen_;
#else
  VSLStreamStatePtr vStream_;
#endif
};


class DataImage;

template <typename XPU, typename DT>
class SparseTensor;

class TensorFormat {
public:
  explicit TensorFormat () { };
  explicit TensorFormat (const libconfig::Config &cfg);
public:
  int rows, cols, chls, nums;
  int numBatch, numField, numClass;
  bool isTrain;
};

template <typename XPU, typename DT>
class Tensor : public XPU {
public:
  explicit Tensor ();
  ~Tensor ();
public:
  void create (const Shape &s, const int did = 0);
  void clear();
  void copy (const Tensor<GPU, DT> &in);
  void copy (const Tensor<CPU, DT> &in);
  Tensor<XPU, DT> segment (const int begin, const int end) const;
  Tensor<XPU, DT> operator[] (const int idx) const { return segment (idx, idx+1);  }
  const Tensor<XPU, DT>& operator= (const Tensor<XPU, DT>& t);
private:
  void mem_alloc();
  void mem_free ();
public:
  void mem_set (const unsigned char a);
  void memcpy_from_gpu (void *ptr);
  void memcpy_from_cpu (void *ptr);
  void memcpy_to_gpu (void *ptr) const;
  void memcpy_to_cpu (void *ptr) const;
public:
  void save (const string file);
  void load (const string file, const int did);
  void show_image ();
  void read_image_data (const TensorFormat &tf, const string &file, const int idx, const Tensor<XPU, DT> &mean);
  void read_image_label (const DataImage &dimg, const string &file, const int idx);
  void read_image (const TensorFormat &tf, const vector<string> &imgList);
public:
  void init (const DT a);
  void init (const Random<XPU> &random, const int rand_method, const DT a=0.f,  const DT b=1.f);
  void im2col_fprop (const Patch &p, Tensor<XPU, DT> &im_col);
  void col2im_bprop (const Patch &p, Tensor<XPU, DT> &im_col);
  void shuffle (const vector<int> &idx);
  void softmax ();
  void add (const DT val);
public:
  void blas_gemm (const bool transA, const bool transB,
    const Tensor<XPU, DT> &A, const Tensor<XPU, DT> &B, DT alpha, DT beta);
  void blas_gemv (const bool transA,
    const Tensor<XPU, DT> &A, const Tensor<XPU, DT> &X, DT alpha, DT beta);
  void sparse_gemv (const bool transA,
    const SparseTensor<XPU, DT> &A, const Tensor<XPU, DT> &X, DT alpha, DT beta);
public:
  void blas_amax (int &idx, DT &val) const;
  void blas_amin (int &idx, DT &val) const;
  void blas_asum (DT &val) const;
  void blas_axpy (const Tensor<XPU, DT> &in, DT alpha);
  void blas_copy_from (const DT *x, const int incx, const int incy);
  void blas_copy_to   (DT *x, const int incx, const int incy) const;
  void blas_sdot (const Tensor<XPU, DT> &in, DT &val) const;
  void blas_nrm2 (DT &val) const;
  void blas_scal (DT alpha);
public:
  void sparse_axpy (const SparseTensor<XPU, DT> &in, DT alpha);
public:
  void blas_vadd (const Tensor<XPU, DT> &A, const Tensor<XPU, DT> &B);
  void blas_vsub (const Tensor<XPU, DT> &A, const Tensor<XPU, DT> &B);
  void blas_vmul (const Tensor<XPU, DT> &A, const Tensor<XPU, DT> &B);
  void blas_vdiv (const Tensor<XPU, DT> &A, const Tensor<XPU, DT> &B);
  void blas_vabs (const Tensor<XPU, DT> &in);
  void blas_vexp (const Tensor<XPU, DT> &in);
  void blas_vinv (const Tensor<XPU, DT> &in);
  void blas_vsqr (const Tensor<XPU, DT> &in);
  void blas_vsqrt(const Tensor<XPU, DT> &in);
public:
  DT reduce_sum () const;
  DT reduce_max () const;
  void reduce_sum  (const Tensor<XPU, DT> &in, const int keepdim);
  void reduce_var  (const Tensor<XPU, DT> &in, const int keepdim);
  void reduce_mean (const Tensor<XPU, DT> &in, const int keepdim);
  void reduce_sum_product (const Tensor<XPU, DT> &ain, const Tensor<XPU, DT> &bin, const int keepdim);
  void bdcast_minus (const Tensor<XPU, DT> &bin, const int keepdim);
  void bdcast_mul   (const Tensor<XPU, DT> &bin, const int keepdim);
  void bdcast_div   (const Tensor<XPU, DT> &bin, const int keepdim);
  void bdcast_minus_product (const Tensor<XPU, DT> &ain, const Tensor<XPU, DT> &bin, const int keepdim);
  void get_mean (Tensor<XPU, DT> &mean) const;
  void sub_mean (const Tensor<XPU, DT> &mean);
public:
  int rows () const { return shape.rows;  }
  int cols () const { return shape.cols;  }
  int chls () const { return shape.chls;  }
  int nums () const { return shape.nums;  }
  int size () const { return shape.size;  }
  size_t size_d () const { return shape.size * sizeof(DT);  }
  void print (const int cnt) const;
#ifdef __CUDACC__
  cudaStream_t   get_copy_stream () const { return dnnctx[did_]->stream_;  }
  cudaStream_t   get_cmpt_stream () const { return dnnctx[did_]->stream_;  }
  cublasHandle_t get_blas_handle () const { return dnnctx[did_]->cublas_;  }
  void setTensor4dDescriptor (cudnnTensorDescriptor_t &desc);
  void setFilter4dDescriptor (cudnnFilterDescriptor_t &desc);
#endif
public:
  Shape shape;
  DT *dptr;
  int did_;
  bool cherry;
};

typedef Tensor<GPU, float>  TensorGPUf;
typedef Tensor<CPU, float>  TensorCPUf;
typedef Tensor<GPU, double> TensorGPUd;
typedef Tensor<CPU, double> TensorCPUd;



template <typename DT>
class DataBuffer {
public:
  explicit DataBuffer () : did_(0), curr_no_(0), lnums_(0) { }
  void reset ();
  void create (const TensorFormat &tf, const int did);
  void page_lock ();
  void page_unlk ();
  void read_tensor (const ParaFileData &pd);
  void read_image  (const TensorFormat &tf, const Tensor<CPU, DT> &mean = Tensor<CPU, DT>());
  void read (const ParaFileData &pd);
  void get_mean (const ParaFileData &pd, const TensorFormat &tf);
  void evaluate (DT &err);
public:
  Tensor<CPU, DT>  inst_;
  Tensor<CPU, DT>  data_;
  Tensor<CPU, DT>  pred_;
  Tensor<CPU, DT> label_;
  DataImage       image_;
  int did_;
  int curr_no_;
  int lnums_, cnums_;
};

template <typename XPU, typename DT>
class DataBatch {
public:
  explicit DataBatch () : did_(0), curr_no_(0) { }
  void reset ();
  void copy (const DataBuffer<DT> &in);
  void send (DataBuffer<DT> &in) const;
  void next (const DataBuffer<DT> &in);
  void rand (const DataBuffer<DT> &in);
  Tensor<XPU, DT>  data_;
  Tensor<XPU, DT>  pred_;
  Tensor<XPU, DT> label_;
  int did_;
  int curr_no_;
  int next_no_;
};

#endif
