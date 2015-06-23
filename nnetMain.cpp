#include <cuda.h>
#include <gflags/gflags.h>

#include "include/tensor.h"
#include "include/nnet.h"

#define XPU GPU

DEFINE_string (config, "config/imagenet56_conv.cfg", "config file");

int main (int argc, char** argv)
{ fLB::FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags (&argc, &argv, true);
  GLogHelper glogh (argv[0], "/home/liuminxu/log/nnetMain.");

//omp_set_num_threads (4);
//mkl_set_num_threads (4);

  libconfig::Config cfg;  cfg.readFile (FLAGS_config.c_str());
  NNetModel<XPU> model;
  model.para_.config (cfg);

  dnnctx.resize (model.para_.num_nnets);
  for (int i = model.para_.min_device; i <= model.para_.max_device; ++i)
  { dnnctx[i] = new XPUCtx (i);
    dnnctx[i]->reset ();
  }
//cuda_set_p2p (model.para_.num_device);

  model.init ();
  model.train ();

//cuda_del_p2p (model.para_.num_device);
//for (int i = 0; i < CUDA_NUM_DEVICE; ++i)
//  delete dnnctx[i];
}
