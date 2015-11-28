#ifndef IMAGE_IO_
#define IMAGE_IO_

#include <sys/stat.h>
#include "../include/util.h"
#include "../include/image.h"
#include "../include/tensor.h"

void mat_2tensor (Mat &src, TensorCPUf &dst)
{ src.convertTo (src, CV_32FC3, 1.f/255);
  for (int i = 0; i < 3; ++i)
    cblas_scopy (src.rows*src.cols, src.ptr<float>()+i, 3, dst[i].dptr, 1);
}

template <>
void TensorCPUf::read_image_data (const TensorFormat &tf, const string &file, const int idx, const TensorCPUf &mean)
{ Mat src = cv::imread (file, 1);
  if (src.data == NULL)
  { LOG (WARNING) << "\timage data invalid\t" << file;
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

  int gap_rows = std::max (0, src.rows - rows());
  int gap_cols = std::max (0, src.cols - cols());
  int y0 = gap_rows / 2;
  int x0 = gap_cols / 2;
  if (tf.isTrain)
  { y0 = gap_rows == 0 ? 0 : rand() % gap_rows;
    x0 = gap_cols == 0 ? 0 : rand() % gap_cols;
  }

  cv::Rect roi (x0, y0, cols(), rows());
  Mat crop = src (roi);
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
