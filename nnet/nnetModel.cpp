#ifndef NNET_MODEL_
#define NNET_MODEL_

#include <algorithm>
#include "../include/nnet.h"

template <typename XPU>
void NNetModel<XPU>::init_model ()
{ train_. resize (para_.num_nnets);
  predt_. resize (para_.num_nnets);
  batch_. resize (para_.num_nnets);
  nodes_. resize (para_.num_nnets);
  layers_.resize (para_.num_nnets);
  optims_.resize (para_.num_nnets);
  trainErr_.resize (para_.num_nnets);
  predtErr_.resize (para_.num_nnets);
  for (int did = para_.min_device; did <= para_.max_device; ++did)
  { cuda_set_device (did);
    mem_free (did);

    nodes_[did].resize (para_.num_nodes);
    nodes_[did][0].create (Shape (
	  para_.tFormat_.rows,
	  para_.tFormat_.cols,
	  para_.tFormat_.chls,
	  para_.tFormat_.nums), did);

    layers_[did].resize (para_.num_layers);
    for (int i = 0; i < para_.num_layers; ++i)
    { ParaLayer &pl = para_.paraLayer_[i];
      layers_[did][i] = create_layer (pl, did, nodes_[did][pl.idxs], nodes_[did][pl.idxd]);
    }

    for (int i = 0; i < para_.num_nodes; ++i)
      nodes_[did][i].shape.print ();
  
    for (int i = 0, j = 0; i < para_.num_layers; ++i)
      if (para_.paraLayer_[i].type == kConvolution)
      { mapLayerWmat_[i] = j;
        layers_[did][i]->init_model ();
        layers_[did][i]->set_optimization (para_.paraWmat_[j], para_.paraBias_[j], optims_[did]);
        j++;
      }

    para_.paraWmat_[0].get_optim_info ();
    para_.paraBias_[0].get_optim_info ();

    batch_[did].data_  = nodes_[did][0];
    batch_[did].pred_  = nodes_[did][para_.num_nodes - 2];
    batch_[did].label_ = nodes_[did][para_.num_nodes - 1];
    batch_[did].set_dnums ();  // TODO
  
    if (para_.model_.if_update)
      load_model (did);
  }
}
template void NNetModel<GPU>::init_model ();
template void NNetModel<CPU>::init_model ();

template <typename XPU>
void NNetModel<XPU>::init_data ()
{ for (int did = para_.min_device; did <= para_.max_device; ++did)
  { cuda_set_device (did);

    train_[did].create (para_.tFormat_, did);
    predt_[did].create (para_.tFormat_, did);
    train_[did].read (para_.train_);
    predt_[did].read (para_.predt_);
    train_[did].read_stats (para_.train_);
    predt_[did].read_stats (para_.predt_);
    train_[did].data_.sub_mean (train_[did].mean_);
    predt_[did].data_.sub_mean (predt_[did].mean_);
#ifdef __CUDACC__
    train_[did].page_lock ();
    predt_[did].page_lock ();
#endif
  }

  if (para_.dataType != "image")
    return;
  metaImage_.init (para_.train_);
  for (int did = para_.min_device; did <= para_.max_device; ++did)
  { train_[did].image_.init (metaImage_, did - para_.min_device, para_.num_device);
    predt_[did].image_.init (para_.predt_);
    train_[did].set_image_lnums ();
    predt_[did].set_image_lnums ();
  }
}
template void NNetModel<GPU>::init_data ();
template void NNetModel<CPU>::init_data ();

template <typename XPU>
void NNetModel<XPU>::mem_free (const int did)
{ cuda_set_device (did);
  for (auto layer : layers_[did])
    delete layer;
  for (auto optim : optims_[did])
    delete optim;
   nodes_[did].clear();
  layers_[did].clear();
  optims_[did].clear();
}
template void NNetModel<GPU>::mem_free (const int did);
template void NNetModel<CPU>::mem_free (const int did);

template <typename XPU>
void NNetModel<XPU>::save_model (const int did)
{ cuda_set_device (did);
  if (did == para_.min_device)
  for (int i = 0; i < para_.num_layers; ++i)
  { char layerid[16];  sprintf (layerid, "%02d", i);
    layers_[did][i]->save_model (para_.model_.path+"_layer_"+layerid);
  }
}
template void NNetModel<GPU>::save_model (const int did);

template <typename XPU>
void NNetModel<XPU>::load_model (const int did)
{ cuda_set_device (did);
  for (int i = 0; i < para_.num_layers; ++i)
  { char layerid[16];  sprintf (layerid, "%02d", i);
    layers_[did][i]->load_model (para_.model_.path+"_layer_"+layerid);
  }
}
template void NNetModel<GPU>::load_model (const int did);



template <typename XPU>
void NNetModel<XPU>::fprop (const int did, const bool is_train)
{ cuda_set_device (did);
  for (size_t i = 0; i < layers_[did].size(); ++i)
    layers_[did][i]->fprop (is_train);
}

template <typename XPU>
void NNetModel<XPU>::bprop (const int did)
{ cuda_set_device (did);
  for (int i = layers_[did].size()-1; i >= 0; --i)
    if (!layers_[did][i]->pl_.isFixed)
      layers_[did][i]->bprop (i != 0);
}

template <typename XPU>
void NNetModel<XPU>::reduce_gmat (const int did)
{ cuda_set_device (did);
  for (int i = optims_[did].size()-1; i >= 0; --i)
    if (!optims_[did][i]->po_.isFixed)
    { for (int r = para_.num_device / 2; r > 0; r /= 2)
      { const int k1  = para_.num_device / r;
        const int pid = did + k1 / 2;
        if (did % k1 != para_.min_device)
        { optims_[did][i]->reduce_notify ();
        } else
        { optims_[did][i]->reduce_wait (*optims_[pid][i]);
          optims_[did][i]->reduce_gmat (*optims_[pid][i]);
        }
      }
      if (did == para_.min_device)
        optims_[did][i]->reduce_scal (1.f/para_.num_device);
    }
}

template <typename XPU>
void NNetModel<XPU>::update_wmat (const int did)
{ cuda_set_device (did);
  for (int i = optims_[did].size()-1; i >= 0; --i)
    if (!optims_[did][i]->po_.isFixed)
    { if (did == para_.min_device)
      { optims_[did][i]->update ();
        optims_[did][i]->accept_notify ();
      } else
      for (int r = 1; r < para_.num_device; r *= 2)
      { const int k1  = para_.num_device / r;
        const int pid = did - k1 / 2;
        if (did % k1 == k1/2)
        { optims_[did][i]->accept_wait (*optims_[pid][i]);
          optims_[did][i]->accept_wmat (*optims_[pid][i]);
          optims_[did][i]->accept_notify ();
        }
      }
    }
}



template <typename XPU>
void NNetModel<XPU>::train ()
{ for (para_.now_round = para_.stt_round; para_.now_round < para_.end_round; para_.now_round++)
#pragma omp parallel for
  for (int did = para_.min_device; did <= para_.max_device; ++did)
  { for (auto optim : optims_[did])
      optim->po_.set_para (para_.now_round, para_.max_round);
    train_epoch (train_[did], batch_[did], did);
  }
#pragma omp parallel for
  for (int did = para_.min_device; did <= para_.max_device; ++did)
     eval_epoch (train_[did], batch_[did], did);
}
template void NNetModel<GPU>::train ();
template void NNetModel<CPU>::train ();

template <typename XPU>
void NNetModel<XPU>::train_epoch (DataBuffer<float> &buffer, DataBatch<XPU, float> &batch, const int did)
{ const int numEvals   = para_.num_evals;
  const int mini_batch = buffer.tf_.nums;
  const int numBatches = buffer.dnums_ / batch .dnums_;
  const int numBuffers = buffer.lnums_ / buffer.dnums_;
  std::thread reader;
  std::random_shuffle (buffer.image_.imgList.begin(), buffer.image_.imgList.end());

  trainErr_[did] = 0.f;
  for (int i = 0; i < numBuffers; ++i)
  { buffer.reset_image_buf (true);
    reader = std::thread (&DataBuffer<float>::read_image_thread, &buffer);

    batch.reset ();
    for (int j = 0; j < numBatches; ++j)
    { buffer.wait_image_buf ((j+1)*mini_batch);
      if (para_.dataType == "image")
        batch.copy (buffer);
      else
        batch.rand (buffer);
      fprop (did, true);
      batch.send (buffer);
//    show_layer (did);
      bprop (did);

      reduce_gmat (did);
      update_wmat (did);
      batch.next (buffer);
    }
    reader.join ();
    buffer.evaluate (trainErr_[did]);

    if ((i+1) % (numBuffers/numEvals) == 0)
    { trainErr_[did] /= i+1;
      eval_epoch (predt_[did], batch, did);
      save_model (did);
      trainErr_[did] *= i+1;
    }
  }
  trainErr_[did] = 0.f;
}

template <typename XPU>
void NNetModel<XPU>::eval_epoch (DataBuffer<float> &buffer, DataBatch<XPU, float> &batch, const int did)
{ const int mini_batch = buffer.tf_.nums;
  const int numBatches = buffer.dnums_ / batch .dnums_;
  const int numBuffers = buffer.lnums_ / buffer.dnums_;
  std::thread reader;
//std::random_shuffle (buffer.image_.imgList.begin(), buffer.image_.imgList.end());

  predtErr_[did] = 0.f;
  for (int i = 0; i < numBuffers; ++i)
  { buffer.reset_image_buf (false);
    reader = std::thread (&DataBuffer<float>::read_image_thread, &buffer);

    batch.reset ();
    for (int j = 0; j < numBatches; ++j)
    { buffer.wait_image_buf ((j+1)*mini_batch);
      batch.copy (buffer);
      fprop (did, false);
      batch.send (buffer);
      batch.next (buffer);
    }
    reader.join ();
    buffer.evaluate (predtErr_[did]);
  }
  char errstr[64];  sprintf (errstr, "\ttrain\t%.4f\tpredt\t%.4f", trainErr_[did], predtErr_[did] / numBuffers);
  LOG (INFO) << "\tGPU  " << did << "\tround  " << para_.now_round << errstr;
}

#endif
