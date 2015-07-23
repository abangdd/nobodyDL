#ifndef IMAGE_IO_
#define IMAGE_IO_

#include <sys/stat.h>
#include "../include/util.h"
#include "../include/image.h"
#include "../include/tensor.h"

using std::cout;
using std::endl;



template <>
void TensorCPUf::show_image ()
{ shape.print ();
  Mat dst;
  if (chls() != 3)
  { int numy = round (sqrt (nums() * chls()) * 3 / 4.);  numy = std::min (numy, nums());
    int numx = round (sqrt (nums() / chls()) * 4 / 3.);  numx = std::max (numx, 1);

    const int numi = std::min (numy*numx, nums());
    dst.create (rows()*numy, cols()*chls()*numx, CV_32F);  //dst = cv::Scalar::all(0);
    for (int i = 0; i < numi; ++i)
      for (int j = 0; j < chls(); ++j)
      { const int x = cols() * (i%numx*chls()+j);
        const int y = rows() * (i/numx);
        Mat roi (dst, cv::Rect (x, y, cols(), rows()));
        for (int k = 0; k < rows(); ++k)
          (*this)[i][j][k].memcpy_to_cpu (roi.ptr<float>(k));
      }
  }
  else
  { int numy = round (sqrt (nums()) * 3 / 4.);  numy = std::min (numy, nums());
    int numx = round (sqrt (nums()) * 4 / 3.);  numx = std::max (numx, 1);

    const int numi = std::min (numy*numx, nums());
    dst.create (rows()*numy, cols()*numx, CV_32FC3);  //dst = cv::Scalar::all(0);
    for (int i = 0; i < numi; ++i)
    { const int x = cols() * (i%numx);
      const int y = rows() * (i/numx);
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
void TensorGPUf::show_image ()
{ TensorCPUf im;  im.create (shape);
  im.copy (*this);
  im.show_image();
}

void pre_process (Mat &src, const TensorFormat &tf, const TensorCPUf &mean, const TensorCPUf &noise)
{ src.convertTo (src, CV_32FC3, 1.f/255);
  mkl_set_num_threads_local (1);
  if (mean.dptr)
    for (int i = 0; i < 3; ++i)
      cblas_saxpy (src.rows*src.cols, -1.f, mean[i].dptr, 1, src.ptr<float>()+i, 3);
//if (tf.isTrain)
//  src += cv::Scalar (noise.dptr[0], noise.dptr[1], noise.dptr[2]);
  if (tf.isTrain && rand() % 2)
    flip (src, src, 1);
/*
  { float angle = -rand() % 21 + 10;
    float scale = 1.f;
    cv::Point2f center (src.cols/2, src.rows/2);
    Mat rotateMat = cv::getRotationMatrix2D (center, angle, scale);
    cv::warpAffine (src, src, rotateMat, src.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT_101);
  }
*/
}

void mat_2tensor (const Mat &src, TensorCPUf &dst)
{ mkl_set_num_threads_local (1);
  for (int i = 0; i < 3; ++i)
    cblas_scopy (src.rows*src.cols, src.ptr<float>()+i, 3, dst[i].dptr, 1);
}

template <>
void TensorCPUf::read_image_data (const TensorFormat &tf, const string &file, const int idx,
  const TensorCPUf &mean, const TensorCPUf &eigvec, const TensorCPUf &eigval, const Random<CPU> &random)
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

  float     crop_ratio= 1.f;
  const int crop_rows = tf.isTrain ? rows() * crop_ratio : rows();
  const int crop_cols = tf.isTrain ? cols() * crop_ratio : cols();
  const int  gap_rows = src.rows - crop_rows;
  const int  gap_cols = src.cols - crop_cols;
  int y0 = gap_rows / 2;
  int x0 = gap_cols / 2;
  if (tf.isTrain)
  { y0 = gap_rows == 0 ? 0 : rand() % gap_rows;
    x0 = gap_cols == 0 ? 0 : rand() % gap_cols;
  }

  cv::Rect roi (x0, y0, crop_cols, crop_rows);
  Mat crop = src (roi);
  if (tf.isTrain)
  { if (crop_ratio < 1)  cv::resize (crop, crop, cv::Size(cols(), rows()), 0, 0, CV_INTER_LINEAR);
    if (crop_ratio > 1)  cv::resize (crop, crop, cv::Size(cols(), rows()), 0, 0, CV_INTER_AREA);
  }

  TensorCPUf noise, coeff;
  if (tf.isTrain)
  { noise.create (eigval.shape);
    coeff.create (eigval.shape);
    coeff.init   (random, GAUSSIAN, 0.f, 1.f);
    coeff.blas_vmul (coeff, eigval);
    noise.blas_gemv (false, eigvec, coeff, 0.1, 0.f);
  }

  TensorCPUf dst = (*this)[idx];
  pre_process (crop, tf, mean, noise);
  mat_2tensor (crop, dst);
}

template <>
void TensorCPUf::read_image_label (const DataImage &dimg, const string &file, const int idx)
{ const string fname = file.substr (dimg.image_path.length ());
  auto got = dimg.label_map.find (fname);
  if (got == dimg.label_map.end ())
    LOG (WARNING) << "\timage label not found\t" << file;
  else
  { (*this)[idx].mem_set (0);
    (*this)[idx].dptr[got->second] = 1.f;
  }
}



void image_resize (const ParaImCvt &para, const string &file_src, const string &file_dst)
{ Mat im_src = cv::imread (file_src, 1);
  if (!im_src.data)
  { LOG (WARNING) << "\timage is invalid\t" << file_src;
    return;
  }
  const int rows = im_src.rows, cols = im_src.cols;
  const float minDim = std::min (para.rows, para.cols);
  const int adaptiveRows = rows >= cols ? round (minDim * rows / cols) : minDim;
  const int adaptiveCols = cols >= rows ? round (minDim * cols / rows) : minDim;
  Mat im_dst;
  cv::resize (im_src, im_dst, cv::Size(adaptiveCols, adaptiveRows), 0, 0, CV_INTER_AREA);
  cv::imwrite(file_dst, im_dst);
}

void image_resize (const ParaImCvt &para, const string &folder_src, const string &folder_dst, const string &suffix)
{ vector<string> srcList;  get_dir_list (folder_src, -1, srcList);
  vector<string> dstList;  dstList = srcList;
  for (size_t i = 0; i < srcList.size(); i++)
  { dstList[i].replace (0, folder_src.length(), folder_dst);
    if (access (dstList[i].c_str(), F_OK) != 0)
      CHECK (mkdir (dstList[i].c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == 0);

    cout << "image resizing\t" << srcList[i] << "\t" << dstList[i] << endl;
    vector<string> fileList;  get_file_list (srcList[i], suffix, fileList);
#pragma omp parallel for
    for (size_t j = 0; j < fileList.size(); j++)
      image_resize (para, srcList[i] + fileList[j], dstList[i] + fileList[j]);
  }
}

int readImage (const libconfig::Config &cfg, const string path, const bool readMask, Mat &src, Mat &mask)
{ const int fixrows = cfg.lookup ("readImage.rows");
  const int fixcols = cfg.lookup ("readImage.cols");
  const string suffix = cfg.lookup ("readImage.suffix");
  const string labelF = cfg.lookup ("data.pathLabel");

  string filename = path.substr (path.rfind("/")+1, path.rfind(suffix)-path.rfind("/")-1);  // 文件名 不含路径和后缀
  string maskpath = labelF + "/" + filename + ".s" + suffix;

  if (suffix == ".jpg")
  { src  = cv::imread (path.c_str());  cout<<path.c_str()<<endl;  }
  if (readMask)  mask = cv::imread (maskpath.c_str());
  else           mask = Mat::zeros (src.size(), CV_8U);

  int rows = src.rows,  cols = src.cols;
  if (rows < 64 || cols < 64)
    return 1;

  rows = src.rows,  cols = src.cols;
  const int adaptiveRows = cols > rows ? (fixrows * rows / cols / 2 * 2) : fixrows;
  const int adaptiveCols = rows > cols ? (fixcols * cols / rows / 2 * 2) : fixcols;
  cv::Size size (adaptiveCols, adaptiveRows);
  resize (src, src, size, cv::INTER_AREA);
  resize (mask,mask,size, cv::INTER_AREA);

  return 0;
}

#endif
