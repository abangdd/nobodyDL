#ifndef NNET_MODEL_
#define NNET_MODEL_

#include <algorithm>
#include "../include/nnet.h"

template <typename XPU>
void NNetModel<XPU>::init_model ()
{ train_. resize (para_.num_nnets);
   test_. resize (para_.num_nnets);
  batch_. resize (para_.num_nnets);
  nodes_. resize (para_.num_nnets);
  layers_.resize (para_.num_nnets);
  optims_.resize (para_.num_nnets);
  for (int did = para_.min_device; did <= para_.max_device; ++did)
  { cuda_set_device (did);
    mem_free (did);

    nodes_[did].resize (para_.num_nodes);
    nodes_[did][0].create                 (para_.shape_src, did);  // TODO
    nodes_[did][para_.num_nodes-1].create (para_.shape_dst, did);  // TODO
  
    layers_[did].resize (para_.num_layers);
    for (int i = 0; i < para_.num_layers; ++i)
    { ParaLayer &pl = para_.paraLayer_[i];
      LOG (INFO) << "\tLayer initializing\t" << para_.paraLayer_[i].get_layer_type();
      layers_[did][i] = create_layer (pl, did, nodes_[did][pl.idxs], nodes_[did][pl.idxd]);
    }

    for (int i = 0; i < para_.num_nodes; ++i)
      nodes_[did][i].shape.print ();
  
    for (int i = 0, j = 0; i < para_.num_layers; ++i)
      if (para_.paraLayer_[i].type == kConvolution || para_.paraLayer_[i].type == kFullConn)
      { layers_[did][i]->init_model ();
        layers_[did][i]->set_optimization (para_.paraWmat_[j], para_.paraBias_[j], optims_[did]);
        layers_[did][i]->get_model_info ();
        j++;
      }

    para_.paraWmat_[para_.num_optims/2 - 1].get_optim_info ();
    para_.paraBias_[para_.num_optims/2 - 1].get_optim_info ();

    batch_[did].data_  = nodes_[did][0];
    batch_[did].pred_  = nodes_[did][para_.num_nodes - 2];
    batch_[did].label_ = nodes_[did][para_.num_nodes - 1];
  
    if (para_.model_.if_update)
    { load_model (did);
    //show_model (did);
    }
  }
}
template void NNetModel<GPU>::init_model ();
template void NNetModel<CPU>::init_model ();

template <typename XPU>
void NNetModel<XPU>::init_data ()
{ for (int did = para_.min_device; did <= para_.max_device; ++did)
  { cuda_set_device (did);

    train_[did].create (para_.tFormat_, did);
     test_[did].create (para_.tFormat_, did);
    train_[did].read (para_.dataTrain_);
     test_[did].read (para_.dataTest_);
    train_[did].read_stats (para_.dataTrain_);
     test_[did].read_stats (para_.dataTest_);
    train_[did].data_.sub_mean (train_[did].mean_);
     test_[did].data_.sub_mean ( test_[did].mean_);
#ifdef __CUDACC__
    train_[did].page_lock ();
     test_[did].page_lock ();
#endif
  }

  dataIm_.init (para_.dataTrain_);
  for (int did = para_.min_device; did <= para_.max_device; ++did)
  { train_[did].image_.init (dataIm_, did - para_.min_device, para_.num_device);
     test_[did].image_.init (para_.dataTest_);
    train_[did].set_image_lnums ();
     test_[did].set_image_lnums ();
  }
}
template void NNetModel<GPU>::init_data ();
template void NNetModel<CPU>::init_data ();

template <typename XPU>
void NNetModel<XPU>::mem_free (const int did)
{ cuda_set_device (did);
  for (size_t i = 0; i < layers_[did].size(); ++i)
    delete layers_[did][i];
  for (size_t i = 0; i < optims_[did].size(); ++i)
    delete optims_[did][i];
   nodes_[did].clear();
  layers_[did].clear();
  optims_[did].clear();
}
template void NNetModel<GPU>::mem_free (const int did);
template void NNetModel<CPU>::mem_free (const int did);

template <typename XPU>
void NNetModel<XPU>::save_model (const int did)
{ cuda_set_device (did);
  if (did != para_.min_device)
    return;
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
void NNetModel<XPU>::show_model (const int did)
{ cuda_set_device (did);
  for (int i = 0; i < para_.num_layers; ++i)
    layers_[did][i]->show_model ();
}
template void NNetModel<GPU>::show_model (const int did);



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
void NNetModel<XPU>::update_wmat (const int did)
{ cuda_set_device (did);
  for (int i = optims_[did].size()-1; i >= 0; --i)
    if (!optims_[did][i]->para_.isFixed)
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
void NNetModel<XPU>::reduce_gmat (const int did)
{ cuda_set_device (did);
  for (int i = optims_[did].size()-1; i >= 0; --i)
    if (!optims_[did][i]->para_.isFixed)
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
void NNetModel<XPU>::train ()
{ for (int r = para_.stt_round; r < para_.end_round; ++r)
#pragma omp parallel for
  for (int did = para_.min_device; did <= para_.max_device; ++did)
  { for (size_t i = 0; i < optims_[did].size(); ++i)
      optims_[did][i]->para_.set_lrate (r, para_.max_round);
    train_epoch (train_[did], did);
  }
}
template void NNetModel<GPU>::train ();
template void NNetModel<CPU>::train ();

template <typename XPU>
void NNetModel<XPU>::train_epoch (DataBuffer<float> &buffer, const int did)
{ const int numEvals   = para_.num_evals;
  const int mini_batch = para_.tFormat_.nums;
  const int numBatches = para_.tFormat_.numBatch;
  const int numBuffers = buffer.lnums_ / buffer.data_.nums();
  std::thread reader;
  std::random_shuffle (buffer.image_.imgList.begin(), buffer.image_.imgList.end());

  for (int i = 0; i < numBuffers; ++i)
  { para_.tFormat_.isTrain = true;
    if (para_.dataTrain_.type == "image")
    { buffer.reset ();
      reader = std::thread (&DataBuffer<float>::read_image_thread, &buffer, std::ref(para_.tFormat_));
    }

    batch_[did].reset ();
    for (int j = 0; j < numBatches; ++j)
    { while (buffer.cnums_ < (j+1)*mini_batch)
        sleep (0.001);
      if (para_.dataTrain_.type == "image")
        batch_[did].copy (buffer);
      else
        batch_[did].rand (buffer);
      fprop (did, true);
      bprop (did);

      reduce_gmat (did);
      update_wmat (did);
      batch_[did].next (buffer);
    }
    reader.join ();

    if ((i+1) % (numBuffers/numEvals) == 0)
    { eval_epoch ( test_[did], did);
      save_model (did);
    }
  }
}

template <typename XPU>
void NNetModel<XPU>::eval_epoch (DataBuffer<float> &buffer, const int did)
{ const int mini_batch = para_.tFormat_.nums;
  const int numBatches = para_.tFormat_.numBatch;
  const int numBuffers = buffer.lnums_ / buffer.data_.nums();
  std::thread reader;

  float test_err = 0.f;
  for (int i = 0; i < numBuffers; ++i)
  { para_.tFormat_.isTrain = false;
    if (para_.dataTest_.type == "image")
    { buffer.reset ();
      reader = std::thread (&DataBuffer<float>::read_image_thread, &buffer, std::ref(para_.tFormat_));
    }

    batch_[did].reset ();
    for (int j = 0; j < numBatches; ++j)
    { while (buffer.cnums_ < (j+1)*mini_batch)
        sleep (0.001);
      batch_[did].copy (buffer);
      fprop (did, false);
      batch_[did].send (buffer);
      batch_[did].next (buffer);
    }
    reader.join ();
    buffer.evaluate (test_err);
  }
  LOG (INFO) << "\tGPU  " << did << "\ttest_err\t" << test_err / numBuffers;
}

#endif
