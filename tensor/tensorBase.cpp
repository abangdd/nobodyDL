#ifndef TENSOR_BASE_
#define TENSOR_BASE_

#include "../include/tensor.h"

using std::ios;

#ifndef __CUDACC__
Shape::Shape () :
  rows(0), cols(0), chls(0), nums(0)
{ set_dims ();
}
Shape::Shape (const int a, const int b, const int c, const int d) :
  rows(a), cols(b), chls(c), nums(d)
{ set_dims ();
}
Shape::Shape (const int a, const int b, const int c, const int d, const int e) :
  rows(a), cols(b), chls(c), nums(d)
{ set_dims ();
  size = e;
}
bool Shape::operator == (const Shape &s) const
{ return (rows == s.rows && cols == s.cols && chls == s.chls && nums == s.nums);
}
bool Shape::operator != (const Shape &s) const
{ return !(*this == s);
}
Shape Shape::section (const int begin, const int end) const
{ const int slices = end - begin;    CHECK (slices >= 1);
  if      (dims == 4)  { CHECK (slices <= nums);  return Shape (rows, cols, chls, slices);  }
  else if (dims == 3)  { CHECK (slices <= chls);  return Shape (rows, cols, slices, nums);  }
  else                 { CHECK (slices <= rows);  return Shape (slices, cols, chls, nums);  }
//else                 { CHECK (shape.dims == 2 && begin == 0 && end == 1);  return shape;  }
}
void Shape::set_dims ()
{ size = rows * cols * chls * nums;
  if      (nums > 1)  dims = 4;
  else if (chls > 1)  dims = 3;
  else if (cols > 1)  dims = 2;
  else                dims = 1;
}
void Shape::set_nums (const int n)
{ CHECK_EQ ((chls * nums) % n, 0);
  chls = chls * nums / n;
  nums = n;
  set_dims ();
}
void Shape::set_chls (const int c)
{ CHECK_EQ ((rows * chls) % c, 0);
  rows = rows * chls / c;
  chls = c;
  set_dims ();
}
void Shape::set_cols (const int c)
{ CHECK_EQ ((rows * cols) % c, 0);
  rows = rows * cols / c;
  cols = c;
  set_dims ();
}
void Shape::re_shape (const int a, const int b, const int c, const int d)
{ CHECK_EQ (a * b * c * d, size);
  rows = a;
  cols = b;
  chls = c;
  nums = d;
  set_dims ();
}
void Shape::print ()
{ char shapestr[64];  sprintf (shapestr, "\tshape\t%d\t%d\t%d\t%d\n", rows, cols, chls, nums);
  LOG (INFO) << shapestr;
}
#endif



#ifndef __CUDACC__
Shape Kernal::get_pack_size (const Shape &in)
{ CHECK (ksize * stride >= 1);
  h_col  = (in.rows + 2 * pad - ksize) / stride + 1;
  w_col  = (in.cols + 2 * pad - ksize) / stride + 1;
  return Shape (ksize * ksize * in.chls, h_col * w_col, 1, in.nums);
}

Shape Kernal::get_pool_size (const Shape &in)
{ CHECK (ksize * stride >= 1);
  h_pool = (in.rows + 2 * pad - ksize) / stride + 1;  // (rows - ksize + stride - 1) / stride + 1
  w_pool = (in.cols + 2 * pad - ksize) / stride + 1;  // (cols - ksize + stride - 1) / stride + 1
  return Shape (h_pool, w_pool, in.chls, in.nums);
}
#endif



template <typename XPU, typename DT>
Tensor<XPU, DT>:: Tensor() : shape(), cherry(false), dptr(NULL), did_(0)
{ }
template <typename XPU, typename DT>
Tensor<XPU, DT>::~Tensor()
{ if (cherry)
    mem_free ();
}
#ifdef __CUDACC__
template TensorGPUf:: Tensor();
template TensorGPUd:: Tensor();
template TensorGPUf::~Tensor();
template TensorGPUd::~Tensor();
#else
template TensorCPUf:: Tensor();
template TensorCPUd:: Tensor();
template TensorCPUf::~Tensor();
template TensorCPUd::~Tensor();
#endif

template <typename XPU, typename DT>
void Tensor<XPU, DT>::create (const Shape &s, const int did)
{ shape = s;
  did_  = did;
  mem_free ();
  mem_alloc();
  cherry = true;
}
#ifdef __CUDACC__
template void TensorGPUf::create (const Shape &s, const int did);
template void TensorGPUd::create (const Shape &s, const int did);
#else
template void TensorCPUf::create (const Shape &s, const int did);
template void TensorCPUd::create (const Shape &s, const int did);
#endif



template <typename XPU, typename DT>
Tensor<XPU, DT> Tensor<XPU, DT>::section (const int begin, const int end) const
{ Tensor<XPU, DT> t;  // TODO
  t.shape = shape.section (begin, end);
  t.dptr  = dptr + t.size() / (end - begin) * begin;
  t.did_  = did_;
  return t;  // TODO
}
#ifndef __CUDACC__
template TensorGPUf TensorGPUf::section (const int begin, const int end) const;
template TensorGPUd TensorGPUd::section (const int begin, const int end) const;
template TensorCPUf TensorCPUf::section (const int begin, const int end) const;
template TensorCPUd TensorCPUd::section (const int begin, const int end) const;
#endif

template <typename XPU, typename DT>
Tensor<XPU, DT>& Tensor<XPU, DT>::operator= (const Tensor<XPU, DT> &t)
{ shape = t.shape;
  dptr  = t.dptr;
  did_  = t.did_;
  cherry = false;
  return *this;
}
#ifndef __CUDACC__
template TensorGPUf& TensorGPUf::operator= (const TensorGPUf &t);
template TensorGPUd& TensorGPUd::operator= (const TensorGPUd &t);
template TensorCPUf& TensorCPUf::operator= (const TensorCPUf &t);
template TensorCPUd& TensorCPUd::operator= (const TensorCPUd &t);
#endif



template <typename XPU, typename DT>
void Tensor<XPU, DT>::mem_alloc ()
{ char pszstr[16];  sprintf (pszstr, "%.2f MB", size_d() / 1e6);
#ifdef __CUDACC__
  LOG_IF (INFO, size_d() > 1e6) << "\tGPU  " << did_ << "  memory required for Tensor\t" << pszstr;
  cuda_malloc ((void**)&dptr, size_d());
#else
  LOG_IF (INFO, size_d() > 1e9) << "\tCPU memory required for Tensor\t" << pszstr;
  dptr = (DT*) malloc (size_d());
#endif
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::mem_free ()
{ 
#ifdef __CUDACC__
  if (dptr)  cuda_check (cudaFree ((void*)dptr));
#else
  if (dptr)  free (dptr);
#endif
  dptr = NULL;
}



#ifdef __CUDACC__
template <typename XPU, typename DT>
void Tensor<XPU, DT>::mem_set (const unsigned char a)
{ cuda_check (cudaMemset ((void*)dptr, a, size_d()));
}
template void TensorGPUf::mem_set (const unsigned char a);
template void TensorGPUd::mem_set (const unsigned char a);
#else
template <typename XPU, typename DT>
void Tensor<XPU, DT>::mem_set (const unsigned char a)
{ memset ((void*)dptr, a, size_d());
}
template void TensorCPUf::mem_set (const unsigned char a);
template void TensorCPUd::mem_set (const unsigned char a);
#endif



#define CPU2GPU cudaMemcpyHostToDevice
#define GPU2CPU cudaMemcpyDeviceToHost
#define CPU2CPU cudaMemcpyHostToHost
#define GPU2GPU cudaMemcpyDeviceToDevice

#ifdef __CUDACC__
template <>
void TensorGPUf::memcpy_from_gpu (void *ptr)
{ cuda_memcpy ((void*)dptr, ptr, size_d(), GPU2GPU);
}
template <>
void TensorGPUd::memcpy_from_gpu (void *ptr)
{ cuda_memcpy ((void*)dptr, ptr, size_d(), GPU2GPU);
}
template <>
void TensorGPUf::memcpy_from_cpu (void *ptr)
{ cuda_memcpy ((void*)dptr, ptr, size_d(), CPU2GPU);
}
template <>
void TensorGPUd::memcpy_from_cpu (void *ptr)
{ cuda_memcpy ((void*)dptr, ptr, size_d(), CPU2GPU);
}
template <>
void TensorCPUf::memcpy_from_gpu (void *ptr)
{ cuda_memcpy ((void*)dptr, ptr, size_d(), GPU2CPU);
}
template <>
void TensorCPUd::memcpy_from_gpu (void *ptr)
{ cuda_memcpy ((void*)dptr, ptr, size_d(), GPU2CPU);
}
#else
template <>
void TensorCPUf::memcpy_from_cpu (void *ptr)
{      memcpy ((void*)dptr, ptr, size_d());
}
template <>
void TensorCPUd::memcpy_from_cpu (void *ptr)
{      memcpy ((void*)dptr, ptr, size_d());
}
#endif

#ifdef __CUDACC__
template <>
void TensorGPUf::memcpy_to_gpu (void *ptr) const
{ cuda_memcpy (ptr, (void*)dptr, size_d(), GPU2GPU);
}
template <>
void TensorGPUf::memcpy_to_cpu (void *ptr) const
{ cuda_memcpy (ptr, (void*)dptr, size_d(), GPU2CPU);
}
template <>
void TensorCPUf::memcpy_to_gpu (void *ptr) const
{ cuda_memcpy (ptr, (void*)dptr, size_d(), CPU2GPU);
}
#else
template <>
void TensorCPUf::memcpy_to_cpu (void *ptr) const
{      memcpy (ptr, (void*)dptr, size_d());
}
#endif



template <typename XPU, typename DT>
void Tensor<XPU, DT>::copy (const Tensor<GPU, DT> &in)
{ CHECK_EQ (size(), in.size());
  memcpy_from_gpu (in.dptr);
}
template <typename XPU, typename DT>
void Tensor<XPU, DT>::copy (const Tensor<CPU, DT> &in)
{ CHECK_EQ (size(), in.size());
  memcpy_from_cpu (in.dptr);
}
#ifdef __CUDACC__
template void TensorGPUf::copy (const TensorGPUf &in);
template void TensorGPUd::copy (const TensorGPUd &in);
template void TensorGPUf::copy (const TensorCPUf &in);
template void TensorGPUd::copy (const TensorCPUd &in);
template void TensorCPUf::copy (const TensorGPUf &in);
template void TensorCPUd::copy (const TensorGPUd &in);
#else
template void TensorCPUf::copy (const TensorCPUf &in);
template void TensorCPUd::copy (const TensorCPUd &in);
#endif

template <typename XPU, typename DT>
void Tensor<XPU, DT>::save (const string file)
{ OFileStream fs (file, ios::binary|ios::trunc);
  fs.write (&shape, sizeof(shape));
  fs.write (dptr, size_d());
}
template <typename XPU, typename DT>
void Tensor<XPU, DT>::load (const string file, const int did)
{ Shape sb = shape;
  IFileStream fs (file, ios::binary);
  fs.read  (&shape, sizeof(shape));
  if (sb != shape && sb.size)
  { LOG (WARNING) << "\tTensor loaded shapes not equal";
    sb.print ();  shape.print ();
  }
  mem_free ();
  mem_alloc();
  fs.read  (dptr, size_d());
  cherry = true;
  did_ = did;
}
#ifdef __CUDACC__
template void TensorGPUf::save (const string file);
template void TensorGPUd::save (const string file);
template void TensorGPUf::load (const string file, const int did);
template void TensorGPUd::load (const string file, const int did);
#else
template void TensorCPUf::save (const string file);
template void TensorCPUd::save (const string file);
template void TensorCPUf::load (const string file, const int did);
template void TensorCPUd::load (const string file, const int did);
#endif

template <typename XPU, typename DT>
void Tensor<XPU, DT>::shuffle (const vector<int> &idx)
{ Tensor<XPU, DT> temp;
  temp.create (shape);
  for (int i = 0; i < nums(); ++ i)
    temp[i].copy ((*this)[idx[i]]);
  copy (temp);
  temp.mem_free ();
}
#ifdef __CUDACC__
template void TensorGPUf::shuffle (const vector<int> &idx);
#else
template void TensorCPUf::shuffle (const vector<int> &idx);
#endif

template <typename XPU, typename DT>
void Tensor<XPU, DT>::softmax ()
{ const DT maxval = reduce_max ();
#pragma omp parallel for
  for (int i = 0; i < size(); ++i)
    dptr[i] = exp (dptr[i] - maxval);
  const DT sumval = reduce_sum ();
    blas_scal ((DT)1./sumval);
}
#ifndef __CUDACC__
template void TensorCPUf::softmax ();
template void TensorCPUd::softmax ();
#endif



template <typename XPU, typename DT>
void Tensor<XPU, DT>::print (const int cnt) const
{ for (int i = 0; i < cnt && i < size(); i++)
    printf ("%.4f\t", dptr[i]);
  printf ("\n");
}
#ifdef __CUDACC__
template void TensorGPUf::print (const int cnt) const;
template void TensorGPUd::print (const int cnt) const;
#else
template void TensorCPUf::print (const int cnt) const;
template void TensorCPUd::print (const int cnt) const;
#endif

#endif
