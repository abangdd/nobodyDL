#ifndef NNET_H_
#define NNET_H_

#include <cudnn.h>
#include <libconfig.h++>
#include "tensor.h"
#include "optimization.h"

#ifdef __CUDACC__
  #define USE_CUDNN true
#endif

using std::max;
using std::min;

enum neuron_t
{ RELU		= 1,
  SIGMOID	= 2,
  TANH		= 3
};

enum pool_t
{ MAX		= 1,
  AVE		= 2
};

enum loss_t
{ ENTROPY	= 1,
  EUCLIDEAN	= 2,
  LOGISTIC	= 3
};

enum layer_t
{ kConvolution	= 1,
  kDropout	= 2,
  kFullConn	= 4,
  kLoss		= 5,
  kNeuron	= 6,
  kPooling	= 8,
  kSoftmax	= 9
};

class ParaLayer {
public:
  explicit ParaLayer () : isLoad(false), isFixed(false) { };
  string get_layer_type ();
#ifdef __CUDACC__
  void setPoolingDescriptor (cudnnPoolingDescriptor_t &desc);
#endif
  int type;
  int idxs, idxd;
  int ksize, pad, stride;
  int flts;
  int random;
  int neuron;
  int pool;
  int loss;
  bool isLoad, isFixed;
  float sigma, bias, dropout;
};



template <typename XPU>
class LayerBase {
public:
  explicit LayerBase (ParaLayer &pl, const int did) : pl_(pl), did_(did) { }
  virtual ~LayerBase () { }
  // is_train the propagation is training or dropout
  virtual void fprop (const bool is_train) = 0;
  virtual void bprop (const bool is_prop_grad) = 0;
public:
  virtual void init_layer () { }
  virtual void init_model () { }
  virtual void save_model (const string file) { }
  virtual void load_model (const string file) { }
  virtual void show_model () { }
  virtual void get_model_info ();
  virtual void set_optimization (ParaOptim &paraWmat, ParaOptim &paraBias, vector<OptimBase<XPU, float>*> &optims) { }
  ParaLayer pl_;
  int did_;
};

#define LAYER_CONSTRUCTOR(layername) \
  explicit layername (ParaLayer &pl, const int did, Tensor<XPU, float> &src, Tensor<XPU, float> &dst) \
  : LayerBase<XPU> (pl, did), pl_(pl), did_(did), src_(src), dst_(dst), rand_(did), alpha(1.), beta(0.) \
  { init_layer ();  }

#define LAYER_FORWARD(layername) \
  template <typename XPU> void layername<XPU>::fprop (const bool is_train)

#define LAYER_BACKPROP(layername) \
  template <typename XPU> void layername<XPU>::bprop (const bool is_prop_grad)

#define LAYER_INIT(layername) \
  template <typename XPU> void layername<XPU>::init_layer ()

#define LAYER_FUNC() \
  void fprop (const bool is_train); \
  void bprop (const bool is_prop_grad); \
  void init_layer ()

#define MODEL_FUNC() \
  void init_model (); \
  void save_model (const string file); \
  void load_model (const string file); \
  void set_optimization (ParaOptim &paraWmat, ParaOptim &paraBias, vector<OptimBase<XPU, float>*> &optims)

#define LAYER_MEMBER \
  ParaLayer pl_; \
  int did_; \
  Tensor<XPU, float> &src_; \
  Tensor<XPU, float> &dst_; \
  Random<XPU> rand_; \
  float alpha, beta

#define MODEL_MEMBER \
  Tensor<XPU, float> drep_,  nrep_; \
  Tensor<XPU, float> wmat_, gwmat_; \
  Tensor<XPU, float> bias_, gbias_; \
  Tensor<XPU, float> scal_, gscal_

template <typename XPU>
class LayerConvolution : public LayerBase<XPU> {
public:
  LAYER_CONSTRUCTOR (LayerConvolution);
  LAYER_FUNC ();
  MODEL_FUNC ();
  void show_model ();
public:
  Patch patch_;
  LAYER_MEMBER;
  MODEL_MEMBER;
private:
  Tensor<XPU, float> tsrc_;
  Tensor<XPU, float> tcol_;
  Tensor<XPU, float> tdst_;
  Tensor<XPU, float> mwmat_, nwmat_, iwmat_;
  int chls_, nums_, flts_, grps_;
  cudnnTensorDescriptor_t srcDesc_, dstDesc_;
  cudnnTensorDescriptor_t biasDesc_;
  cudnnFilterDescriptor_t wmatDesc_;
  cudnnConvolutionDescriptor_t convDesc_;
  cudnnConvolutionFwdAlgo_t algo_;
  size_t worksize;
  void *workspace;
};

template <typename XPU>
class LayerFullConn : public LayerBase<XPU> {
public:
  LAYER_CONSTRUCTOR (LayerFullConn);
  LAYER_FUNC ();
  MODEL_FUNC ();
protected:
  LAYER_MEMBER;
  MODEL_MEMBER;
private:
  Tensor<XPU, float> tsrc_;
  Tensor<XPU, float> tdst_;
  Tensor<XPU, float> mwmat_, nwmat_, iwmat_;
  int dims_, nums_, flts_;
};

template <typename XPU>
class LayerNeuron : public LayerBase<XPU> {
public:
  LAYER_CONSTRUCTOR (LayerNeuron);
  LAYER_FUNC ();
  cudnnActivationMode_t get_activation_type ();
private:
  LAYER_MEMBER;
  cudnnTensorDescriptor_t srcDesc_, dstDesc_;
};

template <typename XPU>
class LayerPooling : public LayerBase<XPU> {
public:
  LAYER_CONSTRUCTOR (LayerPooling);
  LAYER_FUNC ();
public:
  Pool  pool_;
  LAYER_MEMBER;
private:
  Tensor<XPU, float> bsrc_;  // backup
  Tensor<XPU, float> bdst_;  // backup
  cudnnTensorDescriptor_t srcDesc_, dstDesc_;
  cudnnPoolingDescriptor_t  poolDesc_;
};

template <typename XPU>
class LayerDropout : public LayerBase<XPU> {
public:
  LAYER_CONSTRUCTOR (LayerDropout);
  LAYER_FUNC ();
public:
  LAYER_MEMBER;
private:
  Tensor<XPU, float> mask;
  float drop_, scal_;
};

template <typename XPU>
class LayerSoftmax : public LayerBase<XPU> {
public:
  LAYER_CONSTRUCTOR (LayerSoftmax);
  LAYER_FUNC ();
public:
  LAYER_MEMBER;
  Tensor<XPU, float> max_;
  Tensor<XPU, float> sum_;
  Tensor<XPU, float> rep_;
private:
  Tensor<XPU, float> bdst_;  // backup
  int nums_, dims_;
};

template <typename XPU>
class LayerLoss : public LayerBase<XPU> {
public:
  LAYER_CONSTRUCTOR (LayerLoss);
  LAYER_FUNC ();
public:
  LAYER_MEMBER;
private:
  cudnnTensorDescriptor_t srcDesc_, dstDesc_;
  int nums_, dims_;
  float loss_;
};



template<typename XPU>
LayerBase<XPU>* create_layer (ParaLayer &pl, const int did, Tensor<XPU, float> &src, Tensor<XPU, float> &dst);

class ParaNNet {
public:
  explicit ParaNNet () { };
  void config (const libconfig::Config &cfg);
  int get_layer_type (const char *type);
public:
  vector<ParaLayer> paraLayer_;
  vector<ParaOptim> paraWmat_;
  vector<ParaOptim> paraBias_;
  TensorFormat tFormat_;
  ParaFileData dataTrain_;
  ParaFileData dataTest_;
  ParaModel model_;
  Shape shape_src, shape_dst;
  int num_nnets;
  int num_nodes;
  int num_evals;
  int min_device;
  int max_device;
  int num_device;
  int num_layers;
  int num_optims;
  int num_rounds;
  int max_rounds;
};

template <typename XPU>
class NNetModel {
public:
  ~NNetModel ()
  { for (int did = 0; did < para_.num_device; ++did)  mem_free (did);  }
  void mem_free (const int did);
  void init  ();
  void train ();
  void save_model (const int did);
  void load_model (const int did);
  void show_model (const int did);
private:
  void train_epoch (DataBuffer<float> &buffer, const int did);
  void  eval_epoch (DataBuffer<float> &buffer, const int did);
  void fprop (const int did, const bool is_train);
  void bprop (const int did);
  void reduce_gmat (const int did);
  void update_wmat (const int did);
public:
  ParaNNet  para_;
  DataImage dataIm_;
  vector<vector<LayerBase<XPU>*>>        layers_;
  vector<vector<OptimBase<XPU, float>*>> optims_;
  vector<vector<Tensor<XPU, float>>> nodes_;
  vector<DataBatch<XPU, float>>      batch_;
  vector<DataBuffer<float>> train_;
  vector<DataBuffer<float>>  test_;
};

#endif
