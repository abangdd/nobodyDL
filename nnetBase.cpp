#ifndef NNET_BASE_
#define NNET_BASE_

#include "../include/nnet.h"

#ifndef __CUDACC__
string ParaLayer::get_layer_type ()
{ switch (type)
  { case kConvolution	: return "Convolution";
    case kDropout	: return "Dropout";
    case kFullConn	: return "FullConn";
    case kLoss		: return "Loss";
    case kNeuron	: return "Neuron";
    case kPooling	: return "Pooling";
    case kSoftmax	: return "Softmax";
    default		: LOG (FATAL) << "not known layer type";
  }
}

void ParaNNet::config (const libconfig::Config &cfg)
{ tFormat_  = TensorFormat (cfg);
  dataTrain_= ParaFileData (cfg, "traindata");
  dataTest_ = ParaFileData (cfg,  "testdata");

  using namespace libconfig;
  Setting
  &layer_type	= cfg.lookup ("layer.type"),
  &ksize	= cfg.lookup ("layer.ksize"),
  &pad		= cfg.lookup ("layer.pad"),
  &stride	= cfg.lookup ("layer.stride"),
  &flts		= cfg.lookup ("layer.flts"),
  &grps		= cfg.lookup ("layer.grps"),
  &neuron	= cfg.lookup ("layer.neuron_t"),
  &pool		= cfg.lookup ("layer.pool_t"),
  &dropout	= cfg.lookup ("layer.dropout"),
  &loss		= cfg.lookup ("layer.loss_t");

  Setting
  &isLoad	= cfg.lookup ("model.isLoad"),
  &isFixed	= cfg.lookup ("model.isFixed"),
  &isVarN	= cfg.lookup ("model.isVarN"),
  &sigma	= cfg.lookup ("model.sigma"),
  &scale	= cfg.lookup ("model.scale"),
  &bias		= cfg.lookup ("model.bias"),
  &random	= cfg.lookup ("model.rand_t");

  Setting
  &epsW		= cfg.lookup ("optim.epsW"),
  &epsB		= cfg.lookup ("optim.epsB"),
  &wd		= cfg.lookup ("optim.wd");

  int max_fixed_layer = 0;

  paraLayer_.clear();
  for (int i = 0, j = 0, idxn = 0; i < layer_type.getLength(); i++)
  { ParaLayer pl;
    pl.type	= get_layer_type (layer_type[i]);
    pl.idxs	= idxn;
    pl.idxd	= idxn+1;
    pl.ksize	= ksize[i];
    pl.pad	= pad[i];
    pl.stride	= stride[i];
    pl.flts	= flts[i];
    pl.grps	= grps[i];

    pl.neuron	= neuron[i];
    pl.pool	= pool[i];
    pl.dropout	= dropout[i];
    pl.loss	= loss[i];

    if (pl.type == kConvolution || pl.type == kFullConn)
    { pl.isLoad	= isLoad[j];
      pl.isFixed= isFixed[j];
      pl.isVarN = isVarN[j];
      pl.sigma	= sigma[j];
      pl.scale	= scale[j];
      pl.bias	= bias[j];
      pl.random	= random[j];
      j++;
    }

    paraLayer_.push_back (pl);
    idxn++;

    if (pl.neuron > 0)
    { pl.type	= kNeuron;
      pl.idxs	= idxn;
      pl.idxd	= idxn+1;
      paraLayer_.push_back (pl);
      idxn++;
    }
    if (pl.dropout > 0.)
    { pl.type	= kDropout;
      pl.idxs	= idxn;
      pl.idxd	= idxn+1;
      paraLayer_.push_back (pl);
      idxn++;
    }

    if (pl.isFixed)
      max_fixed_layer = pl.idxs;
  }
  for (int i = 0; i < max_fixed_layer; i++)
    paraLayer_[i].isFixed = true;

  paraWmat_.clear();
  paraBias_.clear();
  for (int i = 0; i < isLoad.getLength(); i++)
  { ParaOptim po;
    po.type	= po.get_optim_type (cfg.lookup ("optim.type"));
    po.algo	=                    cfg.lookup ("optim.algo");
    po.isFixed	= isFixed[i];
    po.lr_alpha	= cfg.lookup ("optim.lr_alpha");
    po.lr_last *= NNET_NUM_DEVICES * tFormat_.nums / 128.f;

    po.lr_base	= epsW[i];  po.lr_base *= NNET_NUM_DEVICES;
    po.wd	= wd[i];
    paraWmat_.push_back (po);

    po.lr_base	= epsB[i];  po.lr_base *= NNET_NUM_DEVICES;
    po.wd	= 0.f;
    paraBias_.push_back (po);
  }

  model_.set_para (cfg);

  shape_src = Shape (tFormat_.rows, tFormat_.cols, tFormat_.chls, tFormat_.nums);
  shape_dst = Shape (tFormat_.numClass, 1, 1, tFormat_.nums);

  num_evals  = cfg.lookup ("model.num_evals");
  num_evals /= NNET_NUM_DEVICES;
  num_rounds = cfg.lookup ("model.num_rounds");
  num_layers = paraLayer_.size();
  num_optims = paraWmat_ .size();
  num_nodes  = 0;
  for (int i = 0; i < num_layers; ++i)  // TODO
    num_nodes = std::max (paraLayer_[i].idxd + 1, num_nodes);
}

int ParaNNet::get_layer_type (const char *t)
{ if (!strcmp (t, "conv"	)) return kConvolution;
  if (!strcmp (t, "dropout"	)) return kDropout;
  if (!strcmp (t, "fullc"	)) return kFullConn;
  if (!strcmp (t, "loss"	)) return kLoss;
  if (!strcmp (t, "neuron"	)) return kNeuron;
  if (!strcmp (t, "softmax"	)) return kSoftmax;
  if (!strcmp (t, "pool"	)) return kPooling;
  LOG (FATAL) << "not known layer type";
  return 0;
}
#endif

#ifdef __CUDACC__
void ParaLayer::setPoolingDescriptor (cudnnPoolingDescriptor_t &desc)
{ cuda_check (cudnnSetPooling2dDescriptor (desc, pool == AVE ?
    CUDNN_POOLING_AVERAGE : CUDNN_POOLING_MAX, ksize, ksize, pad, pad, stride, stride));
}

template <>
void TensorGPUf::setTensor4dDescriptor (cudnnTensorDescriptor_t &desc)
{ cuda_check (cudnnSetTensor4dDescriptor (desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, nums(), chls(), rows(), cols()));
}

template <>
void TensorGPUf::setFilter4dDescriptor (cudnnFilterDescriptor_t &desc)
{ cuda_check (cudnnSetFilter4dDescriptor (desc, CUDNN_DATA_FLOAT, nums(), chls(), rows(), cols()));
}
#endif

template<typename XPU>
LayerBase<XPU>* create_layer (ParaLayer &pl, const int did, Tensor<XPU, float> &src, Tensor<XPU, float> &dst)
{ switch (pl.type)
  { case kConvolution	: return new LayerConvolution<XPU>	(pl, did, src, dst);
    case kDropout	: return new LayerDropout<XPU>	(pl, did, src, dst);
    case kFullConn	: return new LayerFullConn<XPU>	(pl, did, src, dst);
    case kLoss		: return new LayerLoss<XPU>	(pl, did, src, dst);
    case kNeuron	: return new LayerNeuron<XPU>	(pl, did, src, dst);
    case kPooling	: return new LayerPooling<XPU>	(pl, did, src, dst);
    case kSoftmax	: return new LayerSoftmax<XPU>	(pl, did, src, dst);
    default		: LOG (FATAL) << "not implemented layer type";
  }
  return NULL;
}
template LayerBase<GPU>* create_layer (ParaLayer &pl, const int did, TensorGPUf &src, TensorGPUf &dst);
template LayerBase<CPU>* create_layer (ParaLayer &pl, const int did, TensorCPUf &src, TensorCPUf &dst);

#endif
