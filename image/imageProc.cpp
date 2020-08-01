#ifndef IMAGE_PROC_
#define IMAGE_PROC_

#include "../include/image.h"

template <> int opencv_data_type<float> (const int chls) { return chls == 3 ? CV_32FC3 : CV_32F; }
template <> int opencv_data_type<double>(const int chls) { return chls == 3 ? CV_64FC3 : CV_64F; }

template <typename DT>
void mat2tensor (const cv::Mat& mat, DT* dptr) {
    const int chls = mat.channels();
    const int area = mat.rows * mat.cols;
    for (int i = 0; i < chls; ++i) {
        const DT* mptr = mat.ptr<DT>()+i;
#pragma omp simd
        for (int j = 0; j < area; ++j)
            dptr[area*i+j] = mptr[chls*j];
    }
}

template <typename DT>
void tensor2mat (const DT* dptr, cv::Mat& mat) {
    const int chls = mat.channels();
    const int area = mat.rows * mat.cols;
    for (int i = 0; i < chls; ++i) {
        DT* mptr = mat.ptr<DT>()+i;
#pragma omp simd
        for (int j = 0; j < area; ++j)
            mptr[chls*j] = dptr[area*i+j];
    }
}

template void mat2tensor (const cv::Mat& src, float*  dptr);
template void mat2tensor (const cv::Mat& src, double* dptr);
template void tensor2mat (const float*  dptr, cv::Mat& src);
template void tensor2mat (const double* dptr, cv::Mat& src);



vector<cv::Point> polygon2points (const vector<float>& poly) {
    vector<cv::Point> pts;
    for (size_t i = 0; i < poly.size()/2; ++i)
        pts.emplace_back (cv::Point (poly[i*2], poly[i*2+1]));
    return pts;
}

vector<cv::Point> polygon2points (const vector<float>& poly, const float hratio, const float wratio) {
    vector<cv::Point> pts;
    for (size_t i = 0; i < poly.size()/2; ++i)
        pts.emplace_back (cv::Point (poly[i*2]*wratio, poly[i*2+1]*hratio));
    return pts;
}

void fill_rlemask (cv::Mat& dst, const vector<size_t>& rlemask) {
    auto dptr = dst.ptr<float>(0);
    for (size_t i = 0, j = 0; i < rlemask.size()/2; ++i) {
        j += rlemask[i*2];
        for (size_t k = 0; k < rlemask[i*2+1]; ++k)
            dptr[j+k] = 1;
        j += rlemask[i*2+1];
    }
}



std::map<int, cv::Mat> masks2mat (const vector<COCOPoly>& coco_poly, const cv::Size& maskSize) {
    std::map<int, cv::Mat> segClass;
    for (auto& anno : coco_poly) {
        if (coco_voc_id_map.find(anno.category_id) == coco_voc_id_map.end())
            continue;
        const int cat_id = coco_voc_id_map.at(anno.category_id);
        const float hratio = maskSize.height / float(anno.size[0]);
        const float wratio = maskSize.width  / float(anno.size[1]);
        vector<vector<cv::Point>> vpts;
        for (auto& poly : anno.polygon)
            vpts.emplace_back (polygon2points (poly, hratio, wratio));
        if (segClass.find(cat_id) == segClass.end())
            segClass[cat_id] = std::move (cv::Mat::zeros (maskSize, CV_32F));
        cv::fillPoly (segClass[cat_id], vpts, cv::Vec3f(1,1,1), cv::LINE_AA);
    }
    return segClass;
}

std::map<int, cv::Mat> masks2mat (const vector<COCOMask>& coco_mask, const cv::Size& maskSize) {
    std::map<int, cv::Mat> segClass;
    for (auto& anno : coco_mask) {
        if (coco_voc_id_map.find(anno.category_id) == coco_voc_id_map.end())
            continue;
        const int cat_id = coco_voc_id_map.at(anno.category_id);
        const cv::Size anno_size (anno.size[1], anno.size[0]);
        if (segClass.find(cat_id) == segClass.end())
            segClass[cat_id] = std::move (cv::Mat::zeros (anno_size, CV_32F));
        fill_rlemask (segClass[cat_id], anno.rlemask);
    }
    for (auto& seg : segClass)
        cv::resize (seg.second, seg.second, maskSize, 0, 0, cv::INTER_NEAREST);
    return segClass;
}

#endif
