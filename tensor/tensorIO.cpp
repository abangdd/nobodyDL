#ifndef TENSOR_IO_
#define TENSOR_IO_

#include "../include/tensor.h"

template <typename XPU, typename DT>
void Tensor<XPU, DT>::save_txt (const string& file) {
    std::stringstream sstr;
    sstr << rows() << "\t" << cols() << "\t" << chls() << "\t" << nums() << std::endl;
    for (int i = 0; i < nums(); ++i) {
        for (int j = 0; j < dims(); ++j)
            sstr << dptr[i*dims()+j] << "\t";
        sstr << std::endl;
    }
    save_file (file, sstr);
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::read_txt (const string& file, const int did) {
    std::stringstream sstr = read_file (file);
    sstr >> shape.rows >> shape.cols >> shape.chls >> shape.nums;
    shape.set_dims();

    create (shape, did);
    for (int i = 0; i < size(); ++i)
        sstr >> dptr[i];
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::save_bin (const string& file) {
    OFileStream fs (file, std::ios::binary|std::ios::trunc);
    fs.write (&shape, sizeof(shape));
    fs.write (dptr, size_d());
}

template <typename XPU, typename DT>
void Tensor<XPU, DT>::load_bin (const string& file, const int did) {
    Shape sb = shape;
    IFileStream fs (file, std::ios::binary);
    fs.read (&shape, sizeof(shape));
    if (sb != shape && sb.size) {
        LOG (WARNING) << "\tTensor loaded shapes not equal";
        sb.print ();
        shape.print ();
    }
    mem_free ();
    mem_alloc();
    fs.read (dptr, size_d());
    did_ = did;
    cherry = true;
}
#ifdef __CUDACC__
template void TensorGPUf::save_bin (const string& file);
template void TensorGPUd::save_bin (const string& file);
template void TensorGPUf::save_txt (const string& file);
template void TensorGPUd::save_txt (const string& file);
template void TensorGPUf::load_bin (const string& file, const int did);
template void TensorGPUd::load_bin (const string& file, const int did);
template void TensorGPUf::read_txt (const string& file, const int did);
template void TensorGPUd::read_txt (const string& file, const int did);
#else
template void TensorCPUf::save_bin (const string& file);
template void TensorCPUd::save_bin (const string& file);
template void TensorCPUf::save_txt (const string& file);
template void TensorCPUd::save_txt (const string& file);
template void TensorCPUf::load_bin (const string& file, const int did);
template void TensorCPUd::load_bin (const string& file, const int did);
template void TensorCPUf::read_txt (const string& file, const int did);
template void TensorCPUd::read_txt (const string& file, const int did);
#endif



template <typename DT>
void TensorBuffer<DT>::save_head_as_dict () {
    const int dims = pred_.dims();
    const int nums = list_size_;  // TODO
    std::stringstream sstr;
    sstr << nums << "\t" << dims << "\n";  // TODO

    save_file (fileData_.anno_path, sstr, std::ios::trunc);
    LOG (INFO) << "\tGPU  " << did_ << "\tsaved head";
}

template <typename DT>
void TensorBuffer<DT>::save_pred_as_dict () {
    const int dims = pred_.dims();
    const int nums = name_.size();  // TODO
    std::stringstream sstr;
    for (int i = 0; i < nums; ++i) {
        sstr << name_[i] << "\t";
        for (int j = 0; j < dims; ++j)
            sstr << pred_.dptr[i*dims+j] << "\t";
        sstr << "\n";
    }

    save_file (fileData_.anno_path, sstr, std::ios::app);
    LOG (INFO) << "\tGPU  " << did_ << "\tsaved pred\t" << nums;
}
#ifndef __CUDACC__
template void TensorBuffer<float>::save_head_as_dict ();
template void TensorBuffer<float>::save_pred_as_dict ();
#endif

#endif
