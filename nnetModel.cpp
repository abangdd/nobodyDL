#ifndef NNET_MODEL_
#define NNET_MODEL_

#include <algorithm>
#include <thread>
#include "../include/nnet.h"

template <typename XPU>
void NNetModel<XPU>::init ()
{ for (int did = 0; did < NNET_NUM_DEVICES; ++did)
  { cuda_set_device (did);
    mem_free (did);
  
    nodes_[did].resize (para_.num_nodes);
    nodes_[did][0].create                 (para_.shape_src, did);  // TODO
    nodes_[did][para_.num_nodes-1].create (para_.shape_dst, did);  // TODO
  
    layers_[did].resize (para_.num_layers);
    for (int i = 0; i < para_.num_layers; ++i)
    { ParaLayer pl = para_.paraLayer_[i];
      LOG (INFO) << "\tLayer initializing\t" << para_.paraLayer_[i].get_layer_type();
      layers_[did][i] = create_layer (pl, did, nodes_[did][pl.idxs], nodes_[did][pl.idxd]);
    }

    for (int i = 0; i < para_.num_nodes; ++i)
      nodes_[did][i].shape.print ();
  
    for (int i = 0, j = 0; i < para_.num_layers; ++i)
      if (para_.paraLayer_[i].type == kConvolution || para_.paraLayer_[i].type == kFullConn)
      { LOG (INFO) << "\tModel initializing\t" << para_.paraLayer_[i].get_layer_type()
          << "\t" << para_.paraLayer_[i].sigma << "\t" << para_.paraLayer_[i].scale;
        layers_[did][i]->init_model ();
        para_.paraWmat_[j].get_optim_info ();
      //para_.paraBias_[j].get_optim_info ();
        layers_[did][i]->set_optimization (para_.paraWmat_[j], para_.paraBias_[j], optims_[did]);
        j++;
      }
  
    batch_[did].data_  = nodes_[did][0];
    batch_[did].pred_  = nodes_[did][para_.num_nodes - 2];
    batch_[did].label_ = nodes_[did][para_.num_nodes - 1];
  
    if (para_.model_.if_update)
    { load_model (did);
    //show_model (did);
    }
    mean_[did].load (para_.dataTrain_.mean, did);
  }
}
template void NNetModel<GPU>::init ();
template void NNetModel<CPU>::init ();

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
void NNetModel<XPU>::fprop (const int did, const bool is_train)
{ cuda_set_device (did);
  for (size_t i = 0; i < layers_[did].size(); ++i)
    layers_[did][i]->fprop (is_train);
}

template <typename XPU>
void NNetModel<XPU>::bprop (const int did)
{ cuda_set_device (did);
  for (size_t i = layers_[did].size(); i > 0; --i)
    if (!layers_[did][i-1]->pl_.isFixed)
      layers_[did][i-1]->bprop (i != 1);
}

template <typename XPU>
void NNetModel<XPU>::update (const int did)
{ cuda_set_device (did);
  for (size_t i = 0; i < optims_[did].size(); ++i)
    if (!optims_[did][i]->para_.isFixed)
    { if (did == 0)
      { optims_[did][i]->update ();
        optims_[did][i]->accept_record ();
      } else
      { optims_[did][i]->accept_wait (*optims_[0][i]);
        optims_[did][i]->accept_wmat (*optims_[0][i]);
      }
    }
}

template <typename XPU>
void NNetModel<XPU>::reduce_gmat (const int did)
{ cuda_set_device (did);
  for (size_t i = 0; i < optims_[did].size(); ++i)
  { for (int r = NNET_NUM_DEVICES / 2; r > 0; r /= 2)
    { const int k1  = NNET_NUM_DEVICES / r;
      const int k2  = k1 * 2;
      const int pid = did + k1 / 2;
      if (k1 > 2 && (did % k2 == k1 || did % k2 == 0 ))
        optims_[did][i]->reduce_wait (*optims_[pid][i]);
      if (k1 > 1 && (did % k2 == k1 || did % k2 == 0 ))
        optims_[did][i]->reduce_gmat (*optims_[pid][i]);
      if (k1 > 1 && (did % k2 == k1))
        optims_[did][i]->reduce_record ();
    }
    if (did == 0)
      optims_[did][i]->reduce_scal (1.f/NNET_NUM_DEVICES);
  }
}



template <typename XPU>
void NNetModel<XPU>::save_model (const int did)
{ cuda_set_device (did);
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
void NNetModel<XPU>::train ()
{ for (int did = 0; did < NNET_NUM_DEVICES; ++did)
  { cuda_set_device (did);

    train_[did].create (para_.tFormat_, did);
     test_[did].create (para_.tFormat_, did);
    train_[did].read (para_.dataTrain_);
     test_[did].read (para_.dataTest_);
    train_[did].data_.sub_mean (mean_[did]);
     test_[did].data_.sub_mean (mean_[did]);
#ifdef __CUDACC__
    train_[did].page_lock ();
     test_[did].page_lock ();
#endif
  }
#pragma omp parallel for
  for (int did = 0; did < NNET_NUM_DEVICES; ++did)
  { for (int r = 0; r < para_.num_rounds; ++r)
    { for (size_t i = 0; i < optims_[did].size(); ++i)
        optims_[did][i]->para_.set_lrate (r, para_.num_rounds);
      train_epoch (train_[did], did);
    }
  }
}
template void NNetModel<GPU>::train ();
template void NNetModel<CPU>::train ();

template <typename XPU>
void NNetModel<XPU>::train_epoch (DataBuffer<float> &buffer, const int did)
{ const int mini_batch = batch_[did].data_.nums();
  const int numBuffers = buffer.lnums_ / buffer.data_.nums();
  const int numBatches = buffer.data_.nums() / mini_batch / NNET_NUM_DEVICES;  // TODO
  const int numEvals = para_.num_evals;
  std::random_shuffle (buffer.image_.img_list.begin(), buffer.image_.img_list.end());

  for (int i = 0; i < numBuffers; ++i)
  { para_.tFormat_.isTrain = true;
    if (para_.dataTrain_.type == "image")
    { buffer.reset ();
      std::thread (&DataBuffer<float>::read_image, &buffer, std::ref(para_.tFormat_), std::ref(mean_[did])).detach ();
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
      update (did);
      batch_[did].next (buffer);
    }

    if ((i+1) % (numBuffers/numEvals) == 0)
    { eval_epoch ( test_[did], did);
    //save_model (did);
    }
  }
}

template <typename XPU>
void NNetModel<XPU>::eval_epoch (DataBuffer<float> &buffer, const int did)
{ const int mini_batch = batch_[did].data_.nums();
  const int numBuffers = buffer.lnums_ / buffer.data_.nums();
  const int numBatches = buffer.data_.nums() / mini_batch;

  float test_err = 0.f;
  for (int i = 0; i < numBuffers; ++i)
  { para_.tFormat_.isTrain = false;
    if (para_.dataTest_.type == "image")
    { buffer.reset ();
      std::thread (&DataBuffer<float>::read_image, &buffer, std::ref(para_.tFormat_), std::ref(mean_[did])).detach ();
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
    buffer.evaluate (test_err);
  }
  LOG (INFO) << "\tGPU  " << did << "\ttest_err\t" << test_err / numBuffers;
}

#endif
