#ifndef NNET_MODEL_
#define NNET_MODEL_

#include <thread>
#include "../include/nnet.h"

template NNetModel<GPU>::NNetModel ();
template NNetModel<CPU>::NNetModel ();

template <typename XPU>
void NNetModel<XPU>::init ()
{ mem_free ();  // TODO

  nodes_.resize (para_.num_nodes);
  nodes_[0].create                 (para_.shape_src);  // TODO
  nodes_[para_.num_nodes-1].create (para_.shape_dst);  // TODO

  layers_.clear();
  for (int i = 0; i < para_.num_layers; ++i)
  { ParaLayer pl = para_.paraLayer_[i];
    LOG (INFO) << "\tLayer initializing\t" << para_.paraLayer_[i].get_layer_type();
    layers_.push_back (create_layer (pl, nodes_[pl.idxs], nodes_[pl.idxd]));
  }

  for (int i = 0; i < para_.num_nodes; ++i)
    nodes_[i].shape.print ();

  for (int i = 0, j = 0; i < para_.num_layers; ++i)
    if (para_.paraLayer_[i].type == kConvolution || para_.paraLayer_[i].type == kFullConn)
    { LOG (INFO) << "\tModel initializing\t" << para_.paraLayer_[i].get_layer_type()
        << "\t" << para_.paraLayer_[i].sigma << "\t" << para_.paraLayer_[i].scale;
      layers_[i]->init_model ();
      para_.paraWmat_[j].get_optim_info ();
    //para_.paraBias_[j].get_optim_info ();
      layers_[i]->set_optimization (para_.paraWmat_[j], para_.paraBias_[j], optims_);
      j++;
    }

  batch_.data_  = nodes_[0];
  batch_.pred_  = nodes_[para_.num_nodes - 2];
  batch_.label_ = nodes_[para_.num_nodes - 1];

  if (para_.model_.if_update)
  { load_model ();
  //show_model ();
  }
  mean_.load (para_.dataTrain_.mean);
}
template void NNetModel<GPU>::init ();
template void NNetModel<CPU>::init ();

template <typename XPU>
void NNetModel<XPU>::mem_free ()
{ for (size_t i = 0; i < layers_.size(); ++i)
    delete layers_[i];
  for (size_t i = 0; i < optims_.size(); ++i)
    delete optims_[i];
   nodes_.clear();
  layers_.clear();
  optims_.clear();
}
template void NNetModel<GPU>::mem_free ();
template void NNetModel<CPU>::mem_free ();

template <typename XPU>
void NNetModel<XPU>::update ()
{ fprop (true);
  for (size_t i = layers_.size(); i > 0; --i)
    if (!layers_[i-1]->pl_.isFixed)
      layers_[i-1]->bprop (i != 1);
  for (size_t i = 0; i < optims_.size(); ++i)
    if (!optims_[i]->para_.isFixed)
      optims_[i]->update ();
  num_iter++;
}

template <typename XPU>
void NNetModel<XPU>::save_model ()
{ for (int i = 0; i < para_.num_layers; ++i)
  { char layerid[16];  sprintf (layerid, "%02d", i);
    layers_[i]->save_model (para_.model_.path+"_layer_"+layerid);
  }
}
template void NNetModel<GPU>::save_model ();

template <typename XPU>
void NNetModel<XPU>::load_model ()
{ for (int i = 0; i < para_.num_layers; ++i)
  { char layerid[16];  sprintf (layerid, "%02d", i);
    layers_[i]->load_model (para_.model_.path+"_layer_"+layerid);
  }
}
template void NNetModel<GPU>::load_model ();

template <typename XPU>
void NNetModel<XPU>::show_model ()
{ for (int i = 0; i < para_.num_layers; ++i)
    layers_[i]->show_model ();
}
template void NNetModel<GPU>::show_model ();



template <typename XPU>
void NNetModel<XPU>::train ()
{ train_.create (para_.tFormat_);
   test_.create (para_.tFormat_);
  train_.read (para_.dataTrain_);
   test_.read (para_.dataTest_);
  train_.data_.sub_mean (mean_);
   test_.data_.sub_mean (mean_);

  for (int r = 0; r < max_round; ++r)
  { train_epoch (train_);
    for (size_t i = 0; i < optims_.size(); ++i)
      optims_[i]->para_.set_lrate (r);
  }
}
template void NNetModel<GPU>::train ();
template void NNetModel<CPU>::train ();

template <typename XPU>
void NNetModel<XPU>::train_epoch (DataBuffer<float> &buffer)
{ const int mini_batch = batch_.data_.nums();
  const int numBuffers = buffer.lnums_ / buffer.data_.nums();
  const int numBatches = buffer.data_.nums() / mini_batch;
  const int numEvals = para_.num_evals;

  for (int i = 0; i < numBuffers; ++i)
  { para_.tFormat_.isTrain = true;
    if (para_.dataTrain_.type == "image")
    { buffer.reset ();
      std::thread (&DataBuffer<float>::read_image, &buffer, std::ref(para_.tFormat_), std::ref(mean_)).detach ();
    }

    batch_.reset ();
    for (int j = 0; j < numBatches; ++j)
    { while (buffer.cnums_ < (j+1)*mini_batch)
        sleep (0.001);
      if (para_.dataTrain_.type == "image")
        batch_.copy (buffer);
      else
        batch_.rand (buffer);
      update ();
      batch_.next (buffer);
    }

    if ((i+1) % (numBuffers/numEvals) == 0)
    { eval_epoch ( test_);
      save_model ();
    }
  }
}

template <typename XPU>
void NNetModel<XPU>::eval_epoch (DataBuffer<float> &buffer)
{ const int mini_batch = batch_.data_.nums();
  const int numBuffers = buffer.lnums_ / buffer.data_.nums();
  const int numBatches = buffer.data_.nums() / mini_batch;

  float test_err = 0.f;
  for (int i = 0; i < numBuffers; ++i)
  { para_.tFormat_.isTrain = false;
    if (para_.dataTest_.type == "image")
    { buffer.reset ();
      std::thread (&DataBuffer<float>::read_image, &buffer, std::ref(para_.tFormat_), std::ref(mean_)).detach ();
    }

    batch_.reset ();
    for (int j = 0; j < numBatches; ++j)
    { while (buffer.cnums_ < (j+1)*mini_batch)
        sleep (0.001);
      batch_.copy (buffer);
      fprop (false);
      batch_.send (buffer);
      batch_.next (buffer);
    }
    buffer.evaluate (test_err);
  }
  LOG (INFO) << "\ttest_err\t" << test_err / numBuffers;
}

#endif
