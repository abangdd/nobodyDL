#ifndef IMAGE_H_
#define IMAGE_H_

#include <cv.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "../include/base.h"
#include "../include/tensor.h"

template <typename DT>
int opencv_data_type (const int chls);

template <typename DT>
void mat2tensor (const cv::Mat& src, DT* dptr);

template <typename DT>
void tensor2mat (const DT* dptr, cv::Mat& dst);

template <typename XPU, typename DT>
void tensor2roi (const Tensor<XPU, DT>& src, cv::Mat& dst);



vector<cv::Point> polygon2points (const vector<float>& poly);

vector<cv::Point> polygon2points (const vector<float>& poly, const float hratio, const float wratio);

cv::Mat contour2mat (const vector<COCOPoly>& anno_coco, const cv::Size& size);

cv::Mat segment2mat (const vector<COCOPoly>& anno_coco, const cv::Size& size);

std::map<int, cv::Mat> masks2mat (const vector<COCOPoly>& coco_poly, const cv::Size& size);

std::map<int, cv::Mat> masks2mat (const vector<COCOMask>& coco_mask, const cv::Size& size);

void fill_coco_anno (const cv::Mat& src, const vector<COCOPoly>& coco_poly, cv::Mat& dst);

void fill_coco_anno (const cv::Mat& src, const vector<COCOMask>& coco_mask, cv::Mat& dst);

template <typename T>
void show_coco_anno (const string dir, const COCOCImage& image, const std::unordered_map<int, vector<T>>& anno_hmap);

void image_resize (const cv::Size& size, const string& srcFile, const string& dstFile);

void image_resize (const cv::Size& size, const string& srcRoot, const string& dstRoot, const std::regex& suffix);

#endif
