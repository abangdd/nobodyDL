#include <gflags/gflags.h>

#include "include/tensor.h"
#include "include/nnet.h"

#define XPU GPU

DEFINE_string (config, "config/coco_seg256_conv_36.cfg", "config file");

int main (int argc, char** argv)
{ fLB::FLAGS_colorlogtostderr = true;
  google::ParseCommandLineFlags (&argc, &argv, true);
  GLogHelper glogh (argv[0], "/home/yourname/log/nnetMain.");
//srand ((unsigned)time(NULL));
//omp_set_num_threads (4);
//mkl_set_num_threads (4);

  libconfig::Config cfg;  cfg.readFile (FLAGS_config.c_str());
  NNetModel<XPU> model;
  model.para_.config (cfg);

  dnnCtx.init(model.para_.min_device, model.para_.max_device);
//cuda_set_p2p (model.para_.num_device);

  model.init_model ();
  model.init_data  ();
  model.train ();

//cuda_del_p2p (model.para_.num_device);
//for (int i = model.para_.min_device; i <= model.para_.max_device; ++i)
//  dnnCtx[i]->release ();
}
