#include "../include/image.h"
#include "../include/tensor.h"

int main (int argc, char** argv) {
    fLB::FLAGS_colorlogtostderr = true;
    google::InstallFailureSignalHandler();

    cv::Size size (256, 256);
    std::regex imgname (".*[(.jpg)(.png)(.JPEG)]");
    image_resize (size, "/mnt/sdc/imagenet/train/", "/mnt/sdb/imagenet/train256/", imgname);
    image_resize (size, "/mnt/sdc/imagenet/val/",   "/mnt/sdb/imagenet/val256/",   imgname);
}
