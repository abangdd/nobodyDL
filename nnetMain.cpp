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

  XPU token;
  if (FLAGS_gpu >= 0)
    cuda_set_device (FLAGS_gpu);
  blas_allocate (token);
//omp_set_num_threads (4);
//mkl_set_num_threads (4);

  libconfig::Config cfg;  cfg.readFile (FLAGS_config.c_str());
  NNetModel<XPU> model;
  model.para_.config (cfg);
  model.init ();
  model.train ();

  blas_release (token);
}
