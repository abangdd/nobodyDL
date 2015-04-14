#include <cuda.h>
#include <omp.h>
#include "include/tensor.h"
#include "include/nnet.h"

#define XPU GPU

DEFINE_string (config, "config/imagenet128.cfg", "config file");
DEFINE_int32  (gpu, 0, "gpu id");

int main (int argc, char** argv)
{ fLB::FLAGS_colorlogtostderr = true;
  google::InstallFailureSignalHandler();
  google::ParseCommandLineFlags (&argc, &argv, true);
  google::InitGoogleLogging (argv[0]);

  dnnctx.resize (CUDA_NUM_DEVICE);
  for (int i = 0; i < CUDA_NUM_DEVICE; ++i)
  { dnnctx[i] = new XPUCtx ();
    dnnctx[i]->set (i);
  }

  if (FLAGS_gpu >= 0)
    cuda_set_device (FLAGS_gpu);
//omp_set_num_threads (4);
//mkl_set_num_threads (4);

  libconfig::Config cfg;  cfg.readFile (FLAGS_config.c_str());
  NNetModel<XPU> model (FLAGS_gpu);
  model.para_.config (cfg);
  model.init ();
  model.train ();

  for (int i = 0; i < CUDA_NUM_DEVICE; ++i)
    delete dnnctx[i];
}
