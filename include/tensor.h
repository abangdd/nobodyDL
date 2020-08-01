#ifndef TENSOR_H_
#define TENSOR_H_

#include "base.h"
#include "expr.h"
#include "xpu.h"

class Shape {
public:
    explicit Shape ();
    explicit Shape (const int a, const int b, const int c, const int d);
    explicit Shape (const int a, const int b, const int c, const int d, const int e);
    explicit Shape (const Shape& s, const vector<int>& keepdim);
    bool operator == (const Shape& s) const;
    bool operator != (const Shape& s) const;
    Shape section (const int begin, const int end) const;
    Shape upsample (const int ksize) const;
    Shape expand_aspp (const int chlp) const;
    Shape expand_area (const int chlt) const;
    Shape expand_chls (const int chlt) const;
    Shape reduce_mask () const;
    void set_dims ();
    void set_chls (const int c);
    void set_cols (const int c);
    void print ();
    void print () const;
    int rows, cols, chls, nums;
    int size, dims;
};

class Kernal {
public:
    explicit Kernal () { }
    explicit Kernal (int a, int b, int c) : krow(a), kcol(b), stride(c) { }
    Shape get_pack_size (const Shape& in);  // 会改变kernal
    Shape get_pool_size (const Shape& in);  // 会改变kernal
    int krow, kcol, pad, stride;
    int h_col, h_pool;
    int w_col, w_pool;
};

void config_head_sizeX_strdX (const Shape& lshape, const Shape& sshape, int& sizeX, int& strdX);
void config_tail_sizeX_strdX (const Shape& lshape, const Shape& sshape, int& sizeX, int& strdX);
void config_side_sizeX_strdX (const Shape& lshape, const Shape& sshape, Shape& tshape);
int get_reduce_type (const Shape& lshape, const Shape& sshape);
int get_reduce_plan (const Shape& lshape, const Shape& sshape, const int sizeT, const int sizeX, unsigned int& blocks, unsigned int& threads);
int get_repeat_plan (const Shape& lshape, const Shape& sshape, const int sizeT, const int sizeX, unsigned int& blocks, unsigned int& threads);



enum rand_t {
    GAUSSIAN = 1,
    UNIFORM = 2
};

template <typename XPU, typename DT>
class Random {
public:
    explicit Random (const int did);
    ~Random();
    void set_seed (int seed);
    void gaussian (DT* data, int size, const DT mu, const DT sigma) const;
    void uniform  (DT* data, int size, const DT a,  const DT b) const;
private:
    int did_;
    VSLStreamStatePtr vStream_;
};

using RandomGPUf = Random<GPU, float>;
using RandomCPUf = Random<CPU, float>;
using RandomGPUd = Random<GPU, double>;
using RandomCPUd = Random<CPU, double>;



template <typename XPU, typename DT>
class SparseTensor;

class BufferFormat {
public:
    explicit BufferFormat () { }
    explicit BufferFormat (const libconfig::Config& cfg);
public:
    int rows, cols, chls, nums;
    int numBatch;
    int numField;
    int numClass;
    bool isTrain;
};

class BufferTrans {
public:
    explicit BufferTrans (const BufferFormat& bfFormat);
public:
    float jitter = 1;
    float padrow = 0, gaprow = 0, offrow = 0.5;
    float padcol = 0, gapcol = 0, offcol = 0.5;
    int rows = 0;  // 原图大小
    int cols = 0;  // 原图大小
    int strd = 4;  // 分割采样
    int flip = 0;  // 水平翻转
};

template <typename XPU, typename DT>
class Tensor : public XPU {
public:
    explicit Tensor () : shape(), cherry(false), dptr(nullptr), did_(0) { }
    Tensor (const Tensor<XPU, DT>& t);
    ~Tensor () { mem_free (); }
public:
    void create (const Shape& s, const int did = 0);
    void create (const Shape& s, DT* sptr, const int did = 0);
    void copy (const Tensor<GPU, DT>& in);
    void copy (const Tensor<CPU, DT>& in);
    Tensor<XPU, DT> mat_view (const int cols) const;
    Tensor<XPU, DT> section (const int begin, const int end) const;
    Tensor<XPU, DT> operator[] (const int idx) const { return section (idx, idx); }
    Tensor<XPU, DT>& operator= (const Tensor<XPU, DT>& t);
private:
    void mem_alloc();
    void mem_free ();
public:
    void mem_set (const unsigned char a);
    void memcpy_from_gpu (void* ptr);
    void memcpy_from_cpu (void* ptr);
    void memcpy_to_gpu (void* ptr) const;
    void memcpy_to_cpu (void* ptr) const;
public:
    void save_txt (const string& file);
    void save_bin (const string& file);
    void read_txt (const string& file, const int did);  // 必须有表头
    void load_bin (const string& file, const int did);
    void show_image (int numc = 0);
    void show_masks (const DT thresh = 0.5, const vector<int>& palette = voc_palette);
public:
    void init (const DT a);
    void init (const Random<XPU, DT>& random, const int method, const DT a=0.f, const DT b=1.f);
    void im2col_fprop (const Kernal& p, Tensor<XPU, DT>& im_col);
    void col2im_bprop (const Kernal& p, Tensor<XPU, DT>& im_col);
    void relu_fprop (const Tensor<XPU, DT>& data);
    void relu_bprop (const Tensor<XPU, DT>& grad);
    void upsample_fprop (const Tensor<XPU, DT>& in, const int ksize);
    void upsample_bprop (const Tensor<XPU, DT>& in, const int ksize);
    void expand_area (const Tensor<XPU, DT>& in, const int chlt);
    void expand_chls (const Tensor<XPU, DT>& in, const int chlt);
    void reduce_mask (const Tensor<XPU, DT>& in, const DT thresh = 0.5);
    void binary_loss (const Tensor<XPU, DT>& pred, const Tensor<XPU, DT>& anno);
    void drop_chls (const Tensor<XPU, DT>& mask, const DT drop);
    void softmax ();
    void sigmoid ();
    void add (const DT val);
public:
    void blas_gemm (const bool transA, const bool transB,
        const Tensor<XPU, DT>& A, const Tensor<XPU, DT>& B, DT alpha, DT beta);
    void blas_gemv (const bool transA,
        const Tensor<XPU, DT>& A, const Tensor<XPU, DT>& X, DT alpha, DT beta);
    void sparse_gemv (const bool transA,
        const SparseTensor<XPU, DT>& A, const Tensor<XPU, DT>& X, DT alpha, DT beta);
public:
    DT blas_asum () const;
    DT blas_nrm2 () const;
    void blas_amax (int& idx, DT& val) const;
    void blas_amin (int& idx, DT& val) const;
    void blas_axpy (const Tensor<XPU, DT>& in, DT alpha);
    void blas_sdot (const Tensor<XPU, DT>& in, DT& val) const;
    void blas_scal (DT alpha);
public:
    void sparse_axpy (const SparseTensor<XPU, DT>& in, DT alpha);
public:
    void blas_vadd (const Tensor<XPU, DT>& A, const Tensor<XPU, DT>& B);
    void blas_vsub (const Tensor<XPU, DT>& A, const Tensor<XPU, DT>& B);
    void blas_vmul (const Tensor<XPU, DT>& A, const Tensor<XPU, DT>& B);
    void blas_vdiv (const Tensor<XPU, DT>& A, const Tensor<XPU, DT>& B);
    void blas_vabs (const Tensor<XPU, DT>& in);
    void blas_vexp (const Tensor<XPU, DT>& in);
    void blas_vinv (const Tensor<XPU, DT>& in);
    void blas_vsqr (const Tensor<XPU, DT>& in);
public:
    DT reduce_sum () const;
    DT reduce_max () const;
    void reduce_sum (const Tensor<XPU, DT>& in);
    void reduce_max (const Tensor<XPU, DT>& in);
    void reduce_min (const Tensor<XPU, DT>& in);
    void reduce_var (const Tensor<XPU, DT>& in);
    void repeat_cpy (const Tensor<XPU, DT>& in);
    void repeat_add (const Tensor<XPU, DT>& in);
    void repeat_sub (const Tensor<XPU, DT>& in);
    void repeat_mul (const Tensor<XPU, DT>& in);
    void repeat_div (const Tensor<XPU, DT>& in);
    void repeat_sub_mean (Tensor<XPU, DT>& mid);
public:
    int rows () const { return shape.rows; }
    int cols () const { return shape.cols; }
    int chls () const { return shape.chls; }
    int nums () const { return shape.nums; }
    int area () const { return shape.rows * shape.cols; }
    int dims () const { return shape.size / shape.nums; }
    int size () const { return shape.size; }
    size_t size_d () const { return shape.size * sizeof(DT); }
    void print (const int cnt) const;
#ifdef __CUDACC__
    cudaStream_t   get_copy_stream () const { return dnnCtx[did_].stream_; }
    cudaStream_t   get_calc_stream () const { return dnnCtx[did_].stream_; }
    cublasHandle_t get_blas_handle () const { return dnnCtx[did_].cublas_; }
    cudnnHandle_t  get_cunn_handle () const { return dnnCtx[did_].cudnn_; }
    void setTensor4dDesc (cudnnTensorDescriptor_t& desc, const int grps = 1) const;
    void setFilter4dDesc (cudnnFilterDescriptor_t& desc, const int grps = 1) const;
#endif
public:
    Shape shape;
    bool cherry;
    DT* dptr;
    int did_;
};

using TensorGPUi = Tensor<GPU, int>;
using TensorCPUi = Tensor<CPU, int>;
using TensorGPUf = Tensor<GPU, float>;
using TensorCPUf = Tensor<CPU, float>;
using TensorGPUd = Tensor<GPU, double>;
using TensorCPUd = Tensor<CPU, double>;



class BatchScheduler {
public:
    void set_size (const int list_size, const int slot_size, const int mini_size);
    void set_parallel (const int degree) { parallel_ = degree; }
    void stop ();
    void sync (const int task);
    void done_add (const int inc = 1);  // CPU完成任务
    bool task_add (const int inc = 1);  // CPU添加任务
    bool proc_get (int& slot, int& task);  // CPU获取任务
    bool sync_get (int& slot, int& task);  // GPU获取任务
    bool wait_done (const int task);  // GPU等待CPU完成
    bool wait_sync (const int task);  // CPU等待GPU同步
private:
    int list_size_ = 0;
    int slot_size_ = 0;
    int mini_size_ = 0;
    int task_pos_ = -1, proc_pos_ = -1;
    int done_pos_ = -1, sync_pos_ = -1;
    int parallel_ = 24;
    bool runnable_ = true;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
};

template <typename DT>
class TensorBuffer {
public:
    ~TensorBuffer () { scheduler_.stop(); }
    void create (const BufferFormat& format, const Shape& src, const Shape& dst, const Shape& val, const int did);
    void read_tensor (const ParaFileData& pd);
    void save_head_as_dict ();
    void save_pred_as_dict ();
    void page_lock ();
    void page_unlk ();
    void read_image_char (BufferTrans& bfTrans, const vector<unsigned char>& bytes, const int idx);
    void read_image_data (BufferTrans& bfTrans, const string& file, const int idx);
    void read_image_anno (BufferTrans& bfTrans, const int anno, const int idx);
    void read_image_anno (BufferTrans& bfTrans, const vector<COCOPoly>& anno, const int idx);
    void read_image_anno (BufferTrans& bfTrans, const vector<COCOMask>& anno, const int idx);
    void proc_image_char (const vector<unsigned char>& bytes, int& slot, int& task);
    void proc_image_data ();
    void proc_image_file ();
    void proc_image_file_slot (const int read_size = 0);
    void evaluate (const int type, DT& error);
public:
    vector<string>  name_;
    Tensor<CPU, DT> data_;
    Tensor<CPU, DT> pred_;
    Tensor<CPU, DT> anno_;
    Tensor<CPU, DT> eval_;
    BufferFormat bfFormat_;
    ParaFileData fileData_;
    BatchScheduler scheduler_;
    int did_ = 0;
    int list_size_ = 0;
    int slot_size_ = 0;
    int mini_size_ = 0;
};

template <typename XPU, typename DT>
class TensorBatch {
public:
    void send_pred (TensorBuffer<DT>& in, const int task) const;
    void copy_data (const TensorBuffer<DT>& in, const int task);
    void rand_data (const TensorBuffer<DT>& in);
    vector<string>  name_;
    Tensor<XPU, DT> data_;
    Tensor<XPU, DT> pred_;
    Tensor<XPU, DT> anno_;
    Tensor<XPU, DT> eval_;
};

template <typename XPU, typename DT>
class TensorDict {
public:
    void save_txt (const string& file);  // 必须有表头
    void read_txt (const string& file);  // 必须有表头
    void save_bin (const string& file, const bool save_name);
    void load_bin (const string& file, const bool read_name);
    void norm_l2 ();
    DT point_dist_l2 (const int i, const int j) const;
public:
    std::unordered_map<string, int> nmap_;
    vector<string>  name_;
    Tensor<XPU, DT> data_;
    int did_ = 0;
};

#endif
