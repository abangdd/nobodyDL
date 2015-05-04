#ifndef TENSOR_H_
#define TENSOR_H_

#include <glog/logging.h>
#include <libconfig.h++>

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
  UNIFORM	= 2,
  XAVIER	= 3
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
  void xavier   (float *data, int size, const int   in, const int out)     const;
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
  explicit Tensor () : shape(), dptr(NULL), did_(0), cherry(false) { }
  ~Tensor ();
public:
  void create (const Shape &s, const int did = 0);
  void clear();
  void peer (const Tensor<GPU, DT> &in);
  void peer (const Tensor<CPU, DT> &in);
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
  void constant (const DT a);
  void random (const Random<XPU> &random, const int rand_method, const DT a=0.f,  const DT b=1.f);
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
