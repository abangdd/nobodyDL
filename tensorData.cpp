#ifndef TENSOR_DATA_
#define TENSOR_DATA_

#include "../include/tensor.h"

#ifndef __CUDACC__
TensorFormat::TensorFormat (const libconfig::Config &cfg) : isTrain(true)
{ rows	= cfg.lookup ("tformat.rows");
  cols	= cfg.lookup ("tformat.cols");
  chls	= cfg.lookup ("tformat.chls");
  nums	= cfg.lookup ("tformat.nums");
  numBatch = cfg.lookup ("tformat.numBatch");
  numClass = cfg.lookup ("tformat.numClass");
};
#endif



template <typename XPU, typename DT>
void Tensor<XPU, DT>::get_mean (Tensor<XPU, DT> &mean) const
{ mean.create ((*this)[0].shape);  mean.mem_set (0);
  for (int i = 0; i < nums(); ++i)
    mean.blas_axpy ((*this)[i], 1.);
  mean.blas_scal (1./nums());
}
#ifdef __CUDACC__
template void TensorGPUf::get_mean (TensorGPUf &mean) const;
template void TensorGPUd::get_mean (TensorGPUd &mean) const;
#else
template void TensorCPUf::get_mean (TensorCPUf &mean) const;
template void TensorCPUd::get_mean (TensorCPUd &mean) const;
#endif

template <typename XPU, typename DT>
void Tensor<XPU, DT>::sub_mean (const Tensor<XPU, DT> &mean)
{ for (int i = 0; i < nums(); ++i)
    (*this)[i].blas_axpy (mean, -1);
}
#ifdef __CUDACC__
template void TensorGPUf::sub_mean (const TensorGPUf &mean);
template void TensorGPUd::sub_mean (const TensorGPUd &mean);
#else
template void TensorCPUf::sub_mean (const TensorCPUf &mean);
template void TensorCPUd::sub_mean (const TensorCPUd &mean);
#endif



#ifdef __CUDACC__
template <typename DT>
void DataBuffer<DT>::page_lock ()
{ cuda_check (cudaHostRegister ( data_.dptr,  data_.size_d(), cudaHostRegisterPortable));
  cuda_check (cudaHostRegister ( pred_.dptr,  pred_.size_d(), cudaHostRegisterPortable));
  cuda_check (cudaHostRegister (label_.dptr, label_.size_d(), cudaHostRegisterPortable));
}
template void DataBuffer<float>::page_lock ();
template <typename DT>
void DataBuffer<DT>::page_unlk ()
{ cuda_check (cudaHostUnregister ( data_.dptr));
  cuda_check (cudaHostUnregister ( pred_.dptr));
  cuda_check (cudaHostUnregister (label_.dptr));
}
template void DataBuffer<float>::page_unlk ();
#else
template <typename DT>
void DataBuffer<DT>::reset ()
{ cnums_ = 0;
   data_.mem_set (0);
   pred_.mem_set (0);
  label_.mem_set (0);
}
template void DataBuffer<float>::reset ();

template <typename DT>
void DataBuffer<DT>::create (const TensorFormat &tf, const int did)
{ Shape dshape (tf.rows, tf.cols, tf.chls, tf.nums*tf.numBatch);
  Shape lshape (      1, tf.numClass,   1, tf.nums*tf.numBatch);
  did_ = did;
   data_.create (dshape, did_);
   pred_.create (lshape, did_);
  label_.create (lshape, did_);
}
template void DataBuffer<float>::create (const TensorFormat &tf, const int did);

template <typename DT>
void DataBuffer<DT>::read_tensor (const ParaFileData &pd)
{  data_.clear ();   data_.load (pd. data, did_);
  label_.clear ();  label_.load (pd.label, did_);
  pred_.create (label_.shape, did_);
  lnums_ = cnums_ = data_.nums();
}
template void DataBuffer<float>::read_tensor (const ParaFileData &pd);

template <typename DT>
void DataBuffer<DT>::read_stats  (const ParaFileData &pd)
{   mean_.load (pd.  mean, 0);
  eigvec_.load (pd.eigvec, 0);
  eigval_.load (pd.eigval, 0);
}
template void DataBuffer<float>::read_stats  (const ParaFileData &pd);

template <>
void DataBuffer<float>::read_image          (const TensorFormat &format)
{ if (curr_no_ + data_.nums() > lnums_)
    curr_no_ = 0;
  cnums_ = 0;
//#pragma omp parallel for
  for (int i = 0; i < data_.nums(); ++i)
  { const int idx = curr_no_ + i;
     data_.read_image_data  (format, image_.img_list[idx], i, mean_, eigvec_, eigval_, rand_);
    label_.read_image_label (image_, image_.img_list[idx], i);
    cnums_ += 1;
  }
  curr_no_ += data_.nums();
//LOG (INFO) << "\timage read\tnumImages = " << data_.nums();
}

template <>
void DataBuffer<float>::read_image_parallel (const TensorFormat &format)
{ if (curr_no_ + data_.nums() > lnums_)
    curr_no_ = 0;
#pragma omp parallel for
  for (int i = 0; i < data_.nums(); ++i)
  { const int idx = curr_no_ + i;
     data_.read_image_data  (format, image_.img_list[idx], i, mean_, eigvec_, eigval_, rand_);
  }
  curr_no_ += data_.nums();
  LOG (INFO) << "\timage read\tnumImages = " << data_.nums();
}

template <typename DT>
void DataBuffer<DT>::read (const ParaFileData &pd)
{ if (pd.type == "tensor")
    read_tensor (pd);
}
template void DataBuffer<float>::read (const ParaFileData &pd);

template <typename DT>
void DataBuffer<DT>::get_mean (const ParaFileData &pd, const TensorFormat &tf)
{ Tensor<CPU, DT> mean_b;
  Tensor<CPU, DT> mean_g;  mean_g.create (data_[0].shape);  mean_g.mem_set (0);

  const int count = lnums_ / data_.nums();
  for (int c = 0; c < count; ++c)
  { if (pd.type == "image")
      read_image (tf);
    data_. get_mean  (mean_b);
    mean_g.blas_axpy (mean_b, 1.);
  }

  mean_g.blas_scal (1./count);
  mean_g.save (pd.mean);
}
template void DataBuffer<float>::get_mean (const ParaFileData &pd, const TensorFormat &tf);

template <typename DT>
void DataBuffer<DT>::sampling (const ParaFileData &pd, const TensorFormat &tf, const int keepdim, Tensor<CPU, DT> &sample)
{ const int nums =  data_.nums();
  const int bats = lnums_ / nums;
  const int cols = 1024;
  const int dims = data_.shape.get_dimsX (keepdim);
  const int strd = data_.shape.get_sizeX (keepdim);
  const int rows = 2147483648 / dims / cols / bats;

  Shape sshape (rows, dims, cols, bats);  sample.create (sshape);
  for (int b = 0; b < bats; ++b)
  { if (pd.type == "image")
      read_image_parallel (tf);
    DT *sptr = sample[b].dptr;
    for (int i = 0; i < rows*cols; ++i)
    { const int idx = rand() % nums;  // i / strd;
      const int pos = rand() % strd;  // i % strd;
      for (int j = 0; j < dims; ++j)
        sptr[i*dims+j] = data_.dptr[(idx*dims+j)*strd+pos];
    }
  }
  sample.shape.re_shape (rows*cols*bats, dims, 1, 1);
//cvar.blas_gemm (true, false, sample, sample, 1, 0);
//cvar.blas_scal (1.f/1000/cols/bats);
}
template void DataBuffer<float>::sampling (const ParaFileData &pd, const TensorFormat &tf, const int keepdim, TensorCPUf &sample);

template <typename DT>
void DataBuffer<DT>::evaluate (DT &err)
{ int eval_idx;
  DT  eval_val;

  DT error = 0;
  for (int i = 0; i < label_.nums(); ++i)
  { pred_[i].blas_amax (eval_idx, eval_val);
    if (label_[i].dptr[eval_idx] != 1)
      error++;
  }
  err += error / label_.nums();
}
template void DataBuffer<float>::evaluate (float &err);
#endif



template <typename XPU, typename DT>
void DataBatch<XPU, DT>::reset ()
{ curr_no_ = 0;
   data_.mem_set (0);
   pred_.mem_set (0);
  label_.mem_set (0);
}
#ifdef __CUDACC__
template void DataBatch<GPU, float>::reset ();
#else
template void DataBatch<CPU, float>::reset ();
#endif

template <typename XPU, typename DT>
void DataBatch<XPU, DT>::copy (const DataBuffer<DT> &in)
{  data_.copy (in. data_.segment (curr_no_, curr_no_+data_.nums()));
  label_.copy (in.label_.segment (curr_no_, curr_no_+data_.nums()));
//DT  label_sum  = label_.reduce_sum ();
//if (label_sum != label_.nums())
//  LOG (ERROR) << "\tbatch labels not correct\t" << label_sum;
}
#ifdef __CUDACC__
template void DataBatch<GPU, float>::copy (const DataBuffer<float> &in);
#else
template void DataBatch<CPU, float>::copy (const DataBuffer<float> &in);
#endif

template <typename XPU, typename DT>
void DataBatch<XPU, DT>::send (DataBuffer<DT> &in) const
{ in. pred_.segment (curr_no_, curr_no_+data_.nums()).copy ( pred_);
}
#ifdef __CUDACC__
template void DataBatch<GPU, float>::send (DataBuffer<float> &in) const;
#else
template void DataBatch<CPU, float>::send (DataBuffer<float> &in) const;
#endif

template <typename XPU, typename DT>
void DataBatch<XPU, DT>::next (const DataBuffer<DT> &in)
{ curr_no_ += data_.nums();
  if (curr_no_ > in.data_.nums())
    curr_no_ = 0;
}
#ifdef __CUDACC__
template void DataBatch<GPU, float>::next (const DataBuffer<float> &in);
#else
template void DataBatch<CPU, float>::next (const DataBuffer<float> &in);
#endif

template <typename XPU, typename DT>
void DataBatch<XPU, DT>::rand (const DataBuffer<DT> &in)
{ const int bsize = data_.nums();
  int begin = ::rand() % in.data_.nums() - bsize;  begin = begin < 0 ? 0 : begin;
   data_.copy (in. data_.segment (begin, begin + bsize));
  label_.copy (in.label_.segment (begin, begin + bsize));
}
#ifdef __CUDACC__
template void DataBatch<GPU, float>::rand (const DataBuffer<float> &in);
#else
template void DataBatch<CPU, float>::rand (const DataBuffer<float> &in);
#endif

#endif
