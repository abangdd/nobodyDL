#ifndef IMAGE_IO_
#define IMAGE_IO_

#include "../include/base.h"
#include "../include/image.h"
#include "../include/tensor.h"

cv::Mat transfer (const cv::Size& cropSize, const BufferTrans& bfTrans, cv::Mat& src) {
    const int crow = round (cropSize.height* bfTrans.jitter);
    const int ccol = round (cropSize.width * bfTrans.jitter);
    const int prow = round (cropSize.height* bfTrans.padrow);
    const int pcol = round (cropSize.width * bfTrans.padcol);

    const int brow = round (std::max (crow - src.rows, 0) * bfTrans.offrow);
    const int bcol = round (std::max (ccol - src.cols, 0) * bfTrans.offcol);
    const int drow = std::max (crow - src.rows - brow, 0);
    const int dcol = std::max (ccol - src.cols - bcol, 0);

    cv::Scalar bgr;
    if (src.type() == CV_8UC3)  // data
        bgr = cv::Scalar (110,113,123);
    else  // anno
        bgr = cv::Scalar (0,0,0);

    if (src.rows < crow || src.cols < ccol)
        cv::copyMakeBorder (src, src, brow, drow, bcol, dcol, cv::BORDER_CONSTANT, bgr);
    else if (bfTrans.padrow > 0 || bfTrans.padcol > 0)
        cv::copyMakeBorder (src, src, prow, prow, pcol, pcol, cv::BORDER_CONSTANT, bgr);

    const int y0 = round ((src.rows - crow) * bfTrans.offrow);
    const int x0 = round ((src.cols - ccol) * bfTrans.offcol);

    cv::Rect roi (x0, y0, ccol, crow);
    cv::Mat crop = src(roi).clone();

    if (bfTrans.flip)
        cv::flip (crop, crop, 1);
    const int interpolation = bfTrans.jitter < 1 ? CV_INTER_CUBIC : CV_INTER_AREA;
    if (bfTrans.jitter != 1)
        cv::resize (crop, crop, cropSize, 0, 0, interpolation);
    return crop;
}

template <typename DT>
void TensorBuffer<DT>::read_image_char (BufferTrans& bfTrans, const vector<unsigned char>& bytes, const int idx) {
    cv::Mat src = cv::imdecode (bytes, CV_LOAD_IMAGE_COLOR);
    if (src.data == nullptr)
        LOG (WARNING) << "\timage data invalid";
    if (src.channels() != bfFormat_.chls)
        LOG (WARNING) << "\timage channels wrong";

    int interpolation = (src.rows*src.cols < bfFormat_.rows*bfFormat_.cols) ? CV_INTER_CUBIC : CV_INTER_AREA;
    cv::Mat crop;
    cv::resize (src, crop, cv::Size(bfFormat_.cols, bfFormat_.rows), 0, 0, interpolation);

    crop.convertTo (crop, CV_32FC3, 1/255.f);
    crop -= cv::Scalar (0.431, 0.444, 0.484);
    mat2tensor (crop, data_[idx].dptr);
}

template <typename DT>
void TensorBuffer<DT>::read_image_data (BufferTrans& bfTrans, const string& file, const int idx) {
    cv::Mat src = cv::imread (file, 1);
    if (src.data == nullptr)
        LOG (WARNING) << "\timage data invalid\t" << file;
    if (src.channels() != data_.chls())
        LOG (WARNING) << "\timage channels wrong\t" << file;

    bfTrans.rows = src.rows;  // TODO
    bfTrans.cols = src.cols;  // TODO

    const cv::Size matCropSize (data_.cols(), data_.rows());
    cv::Mat crop = transfer (matCropSize, bfTrans, src);

    crop.convertTo (crop, CV_32FC3, 1/255.f);
    crop -= cv::Scalar (0.431, 0.444, 0.484);
    mat2tensor (crop, data_[idx].dptr);
}

template <typename DT>
void TensorBuffer<DT>::read_image_anno (BufferTrans& bfTrans, const int anno, const int idx) {
    anno_[idx].mem_set (0);
    if (anno_.area() == 1)
        anno_[idx].dptr[anno] = 1.f;
    else
        anno_[idx][anno].init (1);
}

template <typename DT>
void TensorBuffer<DT>::read_image_anno (BufferTrans& bfTrans, const vector<COCOPoly>& coco_poly, const int idx) {
    const int strd = data_.rows() / anno_.rows();  // TODO
    const cv::Size maskSize (bfTrans.cols/strd, bfTrans.rows/strd);
    const cv::Size cropSize (data_.cols()/strd, data_.rows()/strd);
  //const cv::Size contourSize (bfTrans.cols, bfTrans.rows);
    std::map<int, cv::Mat> masks = masks2mat (coco_poly, maskSize);
  //cv::Mat contour = contour2mat (coco_poly, contourSize);

    anno_[idx].mem_set (0);
    for (auto& kv : masks) {
        const cv::Mat segcrop = transfer (cropSize, bfTrans, kv.second);
        mat2tensor (segcrop, anno_[idx].dptr+segcrop.size().area()*kv.first);
    }
}

template <typename DT>
void TensorBuffer<DT>::read_image_anno (BufferTrans& bfTrans, const vector<COCOMask>& coco_mask, const int idx) {
    const int strd = data_.rows() / anno_.rows();  // TODO
    const cv::Size maskSize (bfTrans.cols/strd, bfTrans.rows/strd);
    const cv::Size cropSize (data_.cols()/strd, data_.rows()/strd);
  //const cv::Size contourSize (bfTrans.cols, bfTrans.rows);
    std::map<int, cv::Mat> masks = masks2mat (coco_mask, maskSize);
  //cv::Mat contour = contour2mat (coco_mask, contourSize);

    anno_[idx].mem_set (0);
    for (auto& kv : masks) {
        const cv::Mat segcrop = transfer (cropSize, bfTrans, kv.second);
        mat2tensor (segcrop, anno_[idx].dptr+segcrop.size().area()*kv.first);
    }
}
template void TensorBuffer<float >::read_image_char (BufferTrans& bfTrans, const vector<unsigned char>& bytes, const int idx);
template void TensorBuffer<double>::read_image_char (BufferTrans& bfTrans, const vector<unsigned char>& bytes, const int idx);
template void TensorBuffer<float >::read_image_data (BufferTrans& bfTrans, const string& file, const int idx);
template void TensorBuffer<double>::read_image_data (BufferTrans& bfTrans, const string& file, const int idx);
template void TensorBuffer<float >::read_image_anno (BufferTrans& bfTrans, const int anno, const int idx);
template void TensorBuffer<double>::read_image_anno (BufferTrans& bfTrans, const int anno, const int idx);
template void TensorBuffer<float >::read_image_anno (BufferTrans& bfTrans, const vector<COCOPoly>& coco_poly, const int idx);
template void TensorBuffer<double>::read_image_anno (BufferTrans& bfTrans, const vector<COCOPoly>& coco_poly, const int idx);
template void TensorBuffer<float >::read_image_anno (BufferTrans& bfTrans, const vector<COCOMask>& coco_mask, const int idx);
template void TensorBuffer<double>::read_image_anno (BufferTrans& bfTrans, const vector<COCOMask>& coco_mask, const int idx);



void image_resize (const cv::Size& size, const string& srcFile, const string& dstFile) {
    cv::Mat src = cv::imread (srcFile, 1);
    if (!src.data) {
        LOG (WARNING) << "\timage is invalid\t" << srcFile;
        return;
    }
    const int rows = src.rows, cols = src.cols;
    float minDim = std::min (size.height, size.width);
    float aRatio = (float)rows / cols;
    int adaRows = rows >= cols ? round (minDim * aRatio) : minDim;
    int adaCols = cols >= rows ? round (minDim / aRatio) : minDim;
    int interpolation = (rows < size.height || cols < size.width) ? CV_INTER_CUBIC : CV_INTER_AREA;
    cv::Mat dst;
    cv::resize (src, dst, cv::Size(adaCols, adaRows), 0, 0, interpolation);
    cv::imwrite(dstFile, dst);
}

#include <experimental/filesystem>

namespace filesystem = std::experimental::filesystem;

string mkdir (const string& srcDir, const string& srcRoot, const string& dstRoot) {
    const string dstDir = dstRoot + srcDir.substr(srcRoot.length());
    if (!filesystem::exists(dstDir))
        CHECK(filesystem::create_directory(dstDir));
    return dstDir;
}

void image_resize (const cv::Size& size, const string& srcRoot, const string& dstRoot, const std::regex& suffix) {
    const vector<string> srcList = get_dir_list (srcRoot);
    for (auto& srcDir : srcList) {
        const string dstDir = mkdir (srcDir, srcRoot, dstRoot);
        const vector<string> fileList = get_file_list (srcDir, suffix);
#pragma omp parallel for
        for (size_t i = 0; i < fileList.size(); ++i) {
            const string srcFile = fileList[i];
            const string dstFile = dstRoot + srcFile.substr(srcRoot.length());
            image_resize (size, srcFile, dstFile);
        }
        LOG (INFO) << "\timage resized from\t" << srcDir << "\tto\t" << dstDir;
    }
}

#endif
