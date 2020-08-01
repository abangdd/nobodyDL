#ifndef NNET_BASE_
#define NNET_BASE_

#include "../include/nnet.h"

#ifndef __CUDACC__
string ParaLayer::get_layer_name () {
    switch (name) {
        case kAct  : return "Act\t"  + std::to_string(isNorm);
        case kConv : return "Conv\t" + std::to_string(std::max(krow,kcol)) + "\t" + std::to_string(grps) + "\t" + std::to_string(hole);
        case kLoss : return "Loss\t" + std::to_string(drop);
        case kPool : return "Pool\t" + std::to_string(drop);
        default : LOG (FATAL) << "unknown layer name";
    }
}

void ParaLayer::get_layer_info () {
    LOG (INFO) << "\tLayer initializing\t" << get_layer_name();
}

void ParaLayer::get_model_info () {
    LOG (INFO) << "\tModel initializing\t" << get_layer_name();
}



int ParaNNet::get_layer_type (const char *t) {
    if (!strcmp (t, "act" )) return kAct;
    if (!strcmp (t, "conv")) return kConv;
    if (!strcmp (t, "loss")) return kLoss;
    if (!strcmp (t, "pool")) return kPool;
    LOG (FATAL) << "unknown layer name";
    return 0;
}

void ParaNNet::config (const libconfig::Config& cfg) {
    format_ = BufferFormat (cfg);  // TODO

    min_device = cfg.lookup ("model.min_device");
    max_device = cfg.lookup ("model.max_device");
    num_device = max_device - min_device + 1;
    num_nnets  = max_device + 1;

    stt_round  = cfg.lookup ("model.stt_round");
    end_round  = cfg.lookup ("model.end_round");
    max_round  = cfg.lookup ("model.max_round");

    using namespace libconfig;
    Setting
    &name = cfg.lookup ("layer.name"),
    &type = cfg.lookup ("layer.type"),
    &krow = cfg.lookup ("layer.krow"),
    &kcol = cfg.lookup ("layer.kcol"),
    &strd = cfg.lookup ("layer.strd"),
    &flts = cfg.lookup ("layer.flts"),
    &grps = cfg.lookup ("layer.grps"),
  //&grid = cfg.lookup ("layer.grid"),
    &hole = cfg.lookup ("layer.hole"),
    &act  = cfg.lookup ("layer.act"),
    &drop = cfg.lookup ("layer.drop");

    Setting
    &isLoad = cfg.lookup ("model.isLoad"),
    &isFixed = cfg.lookup ("model.isFixed");

    float epsW = cfg.lookup ("optim.epsW");
    float epsB = cfg.lookup ("optim.epsB");
    float epsE = cfg.lookup ("optim.epsE");
    float wd = cfg.lookup ("optim.wd");

    paraLayer_.clear();
    for (int i = 0, j = 0, idxn = 0; i < name.getLength(); ++i) {
        ParaLayer pl;
        pl.name = get_layer_type (name[i]);
        pl.type = type[i];
        pl.idxs = idxn;
        pl.idxd = idxn+1;

        pl.krow = krow[i];
        pl.kcol = kcol[i];
        pl.strd = strd[i];
        pl.flts = flts[i];
        pl.grps = grps[i];
      //pl.grid = grid[i];
        pl.hole = hole[i];

      //pl.isNorm = pl.grps == 1 ? true : false;
        pl.act  = act[i];
        pl.drop = drop[i];

        if (pl.name == kConv)
            j++;
        pl.isLoad = isLoad[j-1];
        pl.isFixed = isFixed[j-1];

        if (pl.name == kConv && pl.act >= 1 && pl.isFixed)  // TODO
            pl.isShared = true;
        if (pl.name == kConv && pl.act == 0)  // prior init
            pl.bias = std::log (1.f/pl.flts);

        paraLayer_.push_back (pl);
        idxn++;

        if (pl.act > 0) {
            pl.name = kAct;
            pl.idxs = idxn;
            pl.idxd = idxn+1;
            paraLayer_.push_back (pl);
            idxn++;
        }
    }

    paraWmat_.clear();
    paraBias_.clear();
    paraNorm_.clear();
    
    for (int i = 0; i < isLoad.getLength(); i++) {
        const float lr_multi = num_device * format_.nums / 128.f;
        ParaOptim po;
        po.type = po.get_optim_type (cfg.lookup ("optim.type"));
        po.algo = cfg.lookup ("optim.algo");

        po.lr_base = epsW * lr_multi;
        po.lr_last = epsE * lr_multi;
        po.decay = wd;
        paraWmat_.push_back (po);

        po.lr_base = epsB * lr_multi;
        po.lr_last = epsE * lr_multi;
        po.decay = 0.f;
        paraBias_.push_back (po);

        po.lr_base = epsB * lr_multi;
        po.lr_last = epsE * lr_multi;
        po.decay = 0.f;
        paraNorm_.push_back (po);
    }

    model_ = ParaModel (cfg);

    if (model_.if_train)
        train_ = ParaFileData (cfg, "traindata");
    if (model_.if_infer)
        infer_ = ParaFileData (cfg,  "testdata");

    num_layers = paraLayer_.size();
    num_nodes  = 0;
    for (int i = 0; i < num_layers; ++i)
        num_nodes = std::max (paraLayer_[i].idxd + 1, num_nodes);
}
#else
template <typename XPU, typename DT>
void Tensor<XPU, DT>::setTensor4dDesc (cudnnTensorDescriptor_t& desc, const int grps) const {
    const int sw = 1;
    const int sh = cols() * sw;
    const int sc = rows() * sh;
    const int sn = chls() * sc;
    cuda_check (cudnnSetTensor4dDescriptorEx (desc, cudnn_type<DT>(), nums(), chls()/grps, rows(), cols(), sn, sc, sh, sw));
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::setFilter4dDesc (cudnnFilterDescriptor_t& desc, const int grps) const {
    cuda_check (cudnnSetFilter4dDescriptor (desc, cudnn_type<DT>(), CUDNN_TENSOR_NCHW, nums()/grps, chls(), rows(), cols()));
}

template void TensorGPUf::setTensor4dDesc (cudnnTensorDescriptor_t& desc, const int grps) const;
template void TensorGPUd::setTensor4dDesc (cudnnTensorDescriptor_t& desc, const int grps) const;
template void TensorGPUf::setFilter4dDesc (cudnnFilterDescriptor_t& desc, const int grps) const;
template void TensorGPUd::setFilter4dDesc (cudnnFilterDescriptor_t& desc, const int grps) const;
#endif

template<typename XPU>
std::shared_ptr<LayerBase<XPU>> create_layer (ParaLayer& pl, const int did, Tensor<XPU, float>& src, Tensor<XPU, float>& dst) {
    pl.get_layer_info ();
    switch (pl.name) {
        case kAct  : return std::make_shared<LayerAct <XPU>>(pl, did, src, dst);
        case kConv : return std::make_shared<LayerConv<XPU>>(pl, did, src, dst);
        case kLoss : return std::make_shared<LayerLoss<XPU>>(pl, did, src, dst);
        case kPool : return std::make_shared<LayerPool<XPU>>(pl, did, src, dst);
        default : LOG (FATAL) << "not implemented layer name";
    }
    return nullptr;
}
template std::shared_ptr<LayerBase<GPU>> create_layer (ParaLayer& pl, const int did, TensorGPUf& src, TensorGPUf& dst);
template std::shared_ptr<LayerBase<CPU>> create_layer (ParaLayer& pl, const int did, TensorCPUf& src, TensorCPUf& dst);

#endif
