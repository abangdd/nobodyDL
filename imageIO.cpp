#ifndef IMAGE_IO_
#define IMAGE_IO_

#include <sys/stat.h>
#include "../include/util.h"
#include "../include/image.h"
#include "../include/tensor.h"

template <>
void TensorCPUf::show_image (const int numc)
{ shape.print ();
  Mat dst;
  if (chls() != 3)
  { int numy = round (sqrt (nums() * numc) * 3 / 4.);  numy = std::min (numy, nums());
    int numx = round (sqrt (nums() / numc) * 4 / 3.);  numx = std::max (numx, 1);

    const int numi = std::min (numy*numx, nums());
    dst.create (rows()*numy, cols()*numc*numx, CV_32F);  //dst = cv::Scalar::all(0);
    for (int i = 0; i < numi; ++i)
      for (int j = 0; j < numc; ++j)
      { int x = cols() * (i%numx*numc+j);
        int y = rows() * (i/numx);
        Mat roi (dst, cv::Rect (x, y, cols(), rows()));
        for (int k = 0; k < rows(); ++k)
          (*this)[i][j][k].memcpy_to_cpu (roi.ptr<float>(k));
      }
  }
  else
  { int numy = round (sqrt (nums()) * 3 / 4.);  numy = std::min (numy, nums());
    int numx = round (sqrt (nums()) * 4 / 3.);  numx = std::max (numx, 1);

    const int numi = std::min (numy*numx, nums());
    dst.create (rows()*numy, cols()*numx, CV_32FC3);
    for (int i = 0; i < numi; ++i)
    { int x = cols() * (i%numx);
      int y = rows() * (i/numx);
      Mat roi (dst, cv::Rect (x, y, cols(), rows()));
      for (int j = 0; j < chls(); ++j)
        for (int k = 0; k < rows(); ++k)
          (*this)[i][j][k].blas_copy_to (roi.ptr<float>(k)+j, chls(), 1);
    }
  }
  if (dst.cols > 1600)
    cv::resize (dst, dst, cv::Size (1600, 900));
  normalize (dst, dst, 0, 1, cv::NORM_MINMAX);
  cv::imshow("tensor", dst);
  cv::moveWindow("tensor", 0, 0);
  cv::waitKey(0);  cv::destroyAllWindows();
}

template <>
void TensorGPUf::show_image (const int numc)
{ TensorCPUf im;  im.create (shape);
  im.copy (*this);
  im.show_image (numc);
}

void bgr_normalize (Mat &src)
{ vector<Mat> chlVtr;
  split (src, chlVtr);
  for (int i = 0; i < 3; i++)
    normalize (chlVtr[i], chlVtr[i], 0, 255, cv::NORM_MINMAX);
  merge (chlVtr, src);
}

void bgr_jitter (Mat &src)
{ vector<Mat> chlVtr;
  split (src, chlVtr);
  for (int i = 0; i < 3; ++i)
  { double minVal = 0, maxVal = 0; 
    minMaxLoc (chlVtr[i], &minVal, &maxVal, 0, 0);
    double midVal = (maxVal + minVal) / 2;
    double subVal = (maxVal - minVal) / 2;
    minVal = cv::saturate_cast<uchar> (midVal - subVal * (rand() % 21 + 90) / 100);
    maxVal = cv::saturate_cast<uchar> (midVal + subVal * (rand() % 21 + 90) / 100);
    normalize (chlVtr[i], chlVtr[i], minVal, maxVal, cv::NORM_MINMAX);
  }
  merge (chlVtr, src);
}

void mat_2tensor (Mat &src, TensorCPUf &dst)
{ src.convertTo (src, CV_32FC3, 1.f/255);
  for (int i = 0; i < 3; ++i)
    cblas_scopy (src.rows*src.cols, src.ptr<float>()+i, 3, dst[i].dptr, 1);
}

template <>
void TensorCPUf::read_image_data (const TensorFormat &tf, const string &file, const int idx, const TensorCPUf &mean)
{ Mat src = cv::imread (file, 1);
  if (src.data == NULL)
  { LOG (WARNING) << "\timage is invalid\t" << file;
    return;
  }
  if (src.channels() != chls())
  { LOG (WARNING) << "\timage channels wrong\t" << file;
    return;
  }
  if (src.rows < rows() || src.cols < cols())
  { LOG (WARNING) << "\timage size too small\t" << file;
    return;
  }

  float crop_ratio = 1.f;  // (rand() % 21 + 80) / 100.f;
  int interpolation = crop_ratio < 1 ? CV_INTER_CUBIC : CV_INTER_AREA;
  int crop_rows = tf.isTrain ? rows() * crop_ratio : rows();
  int crop_cols = tf.isTrain ? cols() * crop_ratio : cols();
  int  gap_rows = src.rows - crop_rows;
  int  gap_cols = src.cols - crop_cols;
  int y0 = gap_rows / 2;
  int x0 = gap_cols / 2;
  if (tf.isTrain)
  { y0 = gap_rows == 0 ? 0 : rand() % gap_rows;
    x0 = gap_cols == 0 ? 0 : rand() % gap_cols;
  }

  cv::Rect roi (x0, y0, crop_cols, crop_rows);
  Mat crop = src (roi);
  if (tf.isTrain && crop_ratio != 1)
    cv::resize (crop, crop, cv::Size(cols(), rows()), 0, 0, interpolation);
  if (tf.isTrain && rand() % 2)
    cv::flip   (crop, crop, 1);

  TensorCPUf dst = (*this)[idx];
  mat_2tensor (crop, dst);
  if (mean.dptr)
    dst.blas_axpy (mean, -1);
}

template <>
void TensorCPUf::read_image_label (const MetaImage &dimg, const string &file, const int idx)
{ const string fname = file.substr (dimg.image_path.length ());
  auto got = dimg.label_map.find (fname);
  if (got == dimg.label_map.end ())
    LOG (WARNING) << "\timage label not found\t" << file;
  else
  { (*this)[idx].mem_set (0);
    (*this)[idx].dptr[got->second] = 1.f;
  }
}



void image_resize (const ParaImage &para, const string &fsrc, const string &fdst)
{ Mat isrc = cv::imread (fsrc, 1);
  if (!isrc.data)
  { LOG (WARNING) << "\timage is invalid\t" << fsrc;
    return;
  }
  const int rows = isrc.rows, cols = isrc.cols;
  float minDim = std::min (para.rows, para.cols);
  float aRatio = (float)rows / cols;
  //if (aRatio > 1.33f) aRatio = 1.33f;
  //if (aRatio < 0.75f) aRatio = 0.75f;
  int adaRows = rows >= cols ? round (minDim * aRatio) : minDim;
  int adaCols = cols >= rows ? round (minDim / aRatio) : minDim;
  int interpolation = (rows < para.rows || cols < para.cols) ? CV_INTER_CUBIC : CV_INTER_AREA;
  Mat idst;
  cv::resize (isrc, idst, cv::Size(adaCols, adaRows), 0, 0, interpolation);
  cv::imwrite(fdst, idst);
}

void image_resize (const ParaImage &para, const string &srcRoot, const string &dstRoot, const string &suffix)
{ vector<string> srcList;  get_dir_list (srcRoot, -1, srcList);
  vector<string> dstList;  dstList = srcList;
  for (size_t i = 0; i < srcList.size(); i++)
  { dstList[i].replace (0, srcRoot.length(), dstRoot);
    if (access (dstList[i].c_str(), F_OK) != 0)
      CHECK (mkdir (dstList[i].c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0);

    LOG (INFO) << "\t" << srcList[i] << "\t" << dstList[i];
    vector<string> fileList;  get_file_list (srcList[i], suffix, fileList);
#pragma omp parallel for
    for (size_t j = 0; j < fileList.size(); j++)
      image_resize (para, srcList[i] + fileList[j], dstList[i] + fileList[j]);
  }
}

#endif
