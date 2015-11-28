#include "../include/image.h"
#include "../include/tensor.h"

int main (int argc, char** argv)
{ fLB::FLAGS_colorlogtostderr = true;
  google::InstallFailureSignalHandler();

  ParaImage para (256, 256);
  image_resize (para, "/mnt/sdb/imagenet/train/", "/mnt/sdb/imagenet/train256/", "JPEG");
  image_resize (para, "/mnt/sdb/imagenet/val/",   "/mnt/sdb/imagenet/val256/",   "JPEG");
}
