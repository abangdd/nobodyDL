#ifndef SPARSE_H_
#define SPARSE_H_

#include <libconfig.h++>

#include "xpu.h"
#include "util.h"
#include "tensor.h"
#include "../database/dbMySQL.hpp"

class SparseFormat {
public:
  explicit SparseFormat () { };
  explicit SparseFormat (libconfig::Config &cfg);
public:
  vector<int> col_type;  // -2 ins -1 y
  int format;  // 0 dense 1 sparse
  int numField, numClass;
  int numXFeat, numSFeat;
  int numNumeric, numCombine, numDiscrete;
};



template <typename XPU> void sparse_allocate(XPU a);
template <typename XPU> void sparse_release (XPU a);

#ifdef __CUDACC__
static cusparseHandle_t cusparseHandle;
const char *cuda_get_status (const cusparseStatus_t &status);
#endif

#ifdef __CUDACC__
cusparseOperation_t sparse_get_trans (bool t);
#else
const char*         sparse_get_trans (bool t);
#endif

template <typename XPU, typename DT>
class SparseTensor : public XPU {
public:
  explicit SparseTensor ();
  ~SparseTensor();
public:
  void create (const Shape &s);
  void copy (const SparseTensor<GPU, DT> &in);
  void copy (const SparseTensor<CPU, DT> &in);
private:
  void mem_alloc();
  void mem_free ();
public:
  int rows () const { return shape.rows;  };
  int cols () const { return shape.cols;  };  // 稀疏
  int chls () const { return shape.chls;  };
  int nums () const { return shape.nums;  };
  int size () const { return shape.size;  };
  int size_row () const { return (shape.rows+1) * sizeof(int);  };
  size_t size_idx () const { return  shape.size * sizeof(int);  };
  size_t size_val () const { return  shape.size * sizeof(DT );  };
  void print (const int cnt) const;
public:
  Shape shape;
  bool cherry;
  int *rowPtr;  // length  m+1  
  int *colIdx;  // length  nnz
  DT  *data;    // length  nnz
#ifdef __CUDACC__
  cusparseMatDescr_t descr;  // 放在前面会影响其他成员变量内存
#else
  char descr[6];
#endif
};

typedef SparseTensor<GPU, float>  STensorGPUf;
typedef SparseTensor<CPU, float>  STensorCPUf;
typedef SparseTensor<GPU, double> STensorGPUd;
typedef SparseTensor<CPU, double> STensorCPUd;

template <typename XPU, typename DT>
class SparseBuffer {
public:
  SparseBuffer () : curRow_(0), curCol_(0)  { }
  void create (const int rows, const int cols, const int nnzs);
  void read_db  (const SparseFormat &sf, const ParaDBData &pd, const ParaMySQL &pm);
  void read_str (const SparseFormat &sf, const vector<string> &csv_vtr);
  void save_prediction (const string file);
private:
  void set_data (const int idx, const DT x);
  void set_data (const SparseFormat &sf, const int i, const char *val, char *multi, char *field);
  void parse_db  (const SparseFormat &sf, MYSQL_RES *res);
  void parse_row (const SparseFormat &sf, MYSQL_ROW &row);
  void parse_row (const SparseFormat &sf, const string &csv);
public:
  vector<string>   inst_;
  SparseTensor<XPU, DT> data_;
  Tensor<XPU, DT>  pred_;
  Tensor<XPU, DT> label_;
  int curRow_;
  int curCol_;
};

typedef SparseBuffer<GPU, float>  SBufferGPUf;
typedef SparseBuffer<CPU, float>  SBufferCPUf;
typedef SparseBuffer<GPU, double> SBufferGPUd;
typedef SparseBuffer<CPU, double> SBufferCPUd;

#endif
