#ifndef NNET_MODEL_
#define NNET_MODEL_

#include "../include/nnet.h"

template <typename XPU>
void NNetModel<XPU>::init_model () {
    terr_.resize (para_.num_device);
    perr_.resize (para_.num_device);
    train_.resize (para_.num_device);
    infer_.resize (para_.num_device);
    batch_.resize (para_.num_device);
    nodes_.resize (para_.num_device);
    layers_.resize (para_.num_device);
    optims_.resize (para_.num_device);

    for (int nid = 0; nid < para_.num_device; ++nid) {
        const int did = nid + para_.min_device;
        XPU::set_device (did);

        nodes_[nid].resize (para_.num_nodes);
        nodes_[nid][0].create (Shape (
            para_.format_.rows,
            para_.format_.cols,
            para_.format_.chls,
            para_.format_.nums), did);

        layers_[nid].resize (para_.num_layers);
        for (int i = 0; i < para_.num_layers; ++i) {
            ParaLayer& pl = para_.paraLayer_[i];
            layers_[nid][i] = create_layer (pl, did, nodes_[nid][pl.idxs], nodes_[nid][pl.idxd]);
            if (i == 0)  // data layer
                layers_[nid][i]->set_data (batch_[nid].data_);
            if (i == para_.num_layers-1)  // last layer
                layers_[nid][i]->set_pred (batch_[nid].pred_);
            if (pl.name == kLoss) {  // loss layer
                layers_[nid][i]->set_anno (batch_[nid].anno_);
                layers_[nid][i]->set_eval (batch_[nid].eval_);
            }
        }

        for (int i = 0; i < para_.num_nodes; ++i)
            nodes_[nid][i].shape.print ();
    
        for (int i = 0, j = 0; i < para_.num_layers; ++i) {
            const int name = para_.paraLayer_[i].name;
            if (name == kConv || name == kAct)
                layers_[nid][i]->init_model ();
            if (name == kConv)
                layers_[nid][i]->set_optimization (para_.paraWmat_[j], para_.paraBias_[j], optims_[nid]);
            if (name == kAct)
                layers_[nid][i]->set_optimization (para_.paraNorm_[j], para_.paraNorm_[j], optims_[nid]);
            if (name == kConv)
                j++;
        }

        para_.paraWmat_[0].get_optim_info ();
        para_.paraBias_[0].get_optim_info ();
        para_.paraNorm_[0].get_optim_info ();
    
        if (para_.model_.if_update)
            load_model (nid);
    }
}

template <typename XPU>
void NNetModel<XPU>::init_data () {
    for (int nid = 0; nid < para_.num_device; ++nid) {
        if (para_.train_.data_type == "image")
            train_[nid].fileData_.split_data_anno (para_.train_, nid, para_.num_device);
        if (para_.infer_.data_type == "image")
            infer_[nid].fileData_.split_data_anno (para_.infer_, nid, para_.num_device);
        const int did = nid + para_.min_device;
        XPU::set_device (did);
        train_[nid].create (para_.format_, batch_[nid].data_.shape, batch_[nid].pred_.shape, batch_[nid].eval_.shape, did);
        infer_[nid].create (para_.format_, batch_[nid].data_.shape, batch_[nid].pred_.shape, batch_[nid].eval_.shape, did);
        if (para_.train_.data_type == "tensor")
            train_[nid].read_tensor (para_.train_);
        if (para_.infer_.data_type == "tensor")
            infer_[nid].read_tensor (para_.infer_);
    }
}



template <typename XPU>
void NNetModel<XPU>::save_model (const int nid) {
    XPU::set_device (nid + para_.min_device);
    if (nid == 0)
    for (int i = 0; i < para_.num_layers; ++i) {
        char layerid[16];  sprintf (layerid, "%02d", i);
        layers_[nid][i]->save_model (para_.model_.path+"_layer_"+layerid);
    }
}

template <typename XPU>
void NNetModel<XPU>::load_model (const int nid) {
    XPU::set_device (nid + para_.min_device);
    for (int i = 0; i < para_.num_layers; ++i) {
        char layerid[16];  sprintf (layerid, "%02d", i);
        layers_[nid][i]->load_model (para_.model_.path+"_layer_"+layerid);
    }
}



template <typename XPU>
void NNetModel<XPU>::fprop (const int nid, const bool is_train) {
    XPU::set_device (nid + para_.min_device);
    for (int i = 0; i < para_.num_layers; ++i) {
        layers_[nid][i]->fprop (is_train, false);
    }
}

template <typename XPU>
void NNetModel<XPU>::bprop (const int nid) {
    XPU::set_device (nid + para_.min_device);
    for (int i = para_.num_layers-1; i >= 0; --i)
    if (!layers_[nid][i]->pl_.isFixed) {
        if (i != 0)
            if (layers_[nid][i-1]->pl_.name == kAct)  // TODO
                layers_[nid][i-1]->fprop (true, true);
        layers_[nid][i]->bprop (i != 0);
    }
}

template void NNetModel<GPU>::init_model ();
template void NNetModel<CPU>::init_model ();
template void NNetModel<GPU>::init_data ();
template void NNetModel<CPU>::init_data ();

template void NNetModel<GPU>::save_model (const int nid);
template void NNetModel<CPU>::save_model (const int nid);
template void NNetModel<GPU>::load_model (const int nid);
template void NNetModel<CPU>::load_model (const int nid);

template void NNetModel<GPU>::fprop (const int nid, const bool is_train);
template void NNetModel<CPU>::fprop (const int nid, const bool is_train);
template void NNetModel<GPU>::bprop (const int nid);
template void NNetModel<CPU>::bprop (const int nid);

#endif
