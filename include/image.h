#ifndef IMAGE_H_
#define IMAGE_H_

#include <cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../include/util.h"

using std::cout;
using std::endl;
using cv::Mat;

void image_resize (const ParaImage &para, const string &srcFile, const string &dstFile);

void image_resize (const ParaImage &para, const string &srcRoot, const string &dstRoot, const string &suffix);


#endif
