#include "../include/image.h"
#include "../include/tensor.h"

int main (int argc, char** argv) {
    fLB::FLAGS_colorlogtostderr = true;
    google::InstallFailureSignalHandler();

    cv::Size size (256, 256);
    std::regex imgname (".*[(.jpg)(.png)(.JPEG)]");
    image_resize (size, "/mnt/sdb/coco/train2017/", "/mnt/sdb/coco/train256/", imgname);
    image_resize (size, "/mnt/sdb/coco/val2017/",   "/mnt/sdb/coco/val256/",   imgname);
}
