#include "../include/tensor.h"
#include "../include/image.h"

int main (int argc, char** argv)
{ fLB::FLAGS_colorlogtostderr = true;
  google::InstallFailureSignalHandler();

  libconfig::Config cfg;  cfg.readFile ("config/imagenet224_conv_16.cfg");

  TensorFormat tf (cfg);  tf.isTrain = false;

  DataBuffer<float> buffer;
  buffer.create (tf, 0);

  ParaFileData pd (cfg, "traindata");
  buffer.read (pd);
  buffer.image_.init (pd);
  buffer.set_image_lnums ();
  buffer.get_mean (pd);
}
