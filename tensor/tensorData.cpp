#ifndef TENSOR_DATA_
#define TENSOR_DATA_

#include "../include/tensor.h"

#ifndef __CUDACC__
BufferFormat::BufferFormat (const libconfig::Config& cfg) : isTrain(true) {
    rows = cfg.lookup ("tformat.rows");
    cols = cfg.lookup ("tformat.cols");
    chls = cfg.lookup ("tformat.chls");
    nums = cfg.lookup ("tformat.nums");
    numBatch = cfg.lookup ("tformat.numBatch");
    numClass = cfg.lookup ("tformat.numClass");
};

BufferTrans::BufferTrans (const BufferFormat& bfFormat) {
    if (bfFormat.isTrain) {
        flip = rand() % 2;
        jitter = (rand() % 3 + 4)/4.f;
        offrow = (rand() % 10000)/10000.f;
        offcol = (rand() % 10000)/10000.f;
    }
    if (bfFormat.isTrain && jitter > 1.5) {
        padrow = jitter / 8;
        padcol = jitter / 8;
    }
}



void BatchScheduler::set_size (const int list_size, const int slot_size, const int mini_size) {
    list_size_ = list_size;
    slot_size_ = slot_size;
    mini_size_ = mini_size;
}

void BatchScheduler::stop () {
    std::lock_guard<std::mutex> taskLock (mtx_);
    runnable_ = false;
    cv_.notify_all();
}

void BatchScheduler::sync (const int task) {
    std::lock_guard<std::mutex> taskLock (mtx_);
    sync_pos_ = task;
    cv_.notify_all();
}

void BatchScheduler::done_add (const int inc) {
    std::lock_guard<std::mutex> taskLock (mtx_);
    done_pos_ += inc;
    cv_.notify_all();
}

bool BatchScheduler::task_add (const int inc) {
    std::unique_lock<std::mutex> taskLock (mtx_);
    while ((runnable_ && task_pos_ + inc >= list_size_ && sync_pos_ < task_pos_)
        || (runnable_ && task_pos_ >= done_pos_ + parallel_ && inc == 1))  // inference concurrency
        cv_.wait(taskLock);
    if (!runnable_)  // quit
        return false;
    if (task_pos_ + inc >= list_size_)  // recycling when sync_pos_ == task_pos_
        task_pos_ = proc_pos_ = done_pos_ = sync_pos_ = -1;
    task_pos_ += inc;
    cv_.notify_all();
    return true;
}

bool BatchScheduler::proc_get (int& slot, int& task) {
    std::unique_lock<std::mutex> taskLock (mtx_);
    while (runnable_ && proc_pos_ >= task_pos_)
        cv_.wait(taskLock);
    if (!runnable_)  // quit
        return false;
    proc_pos_ += 1;
    slot = proc_pos_ / slot_size_ * slot_size_;
    task = proc_pos_;
    cv_.notify_one();
    return true;
}

bool BatchScheduler::sync_get (int& slot, int& task) {
    std::unique_lock<std::mutex> taskLock (mtx_);
    while (runnable_ && sync_pos_ >= done_pos_)
        cv_.wait(taskLock);
    if (!runnable_)  // quit
        return false;
    const int sync_done = std::min (done_pos_, sync_pos_ + mini_size_);
    const int sync_ceil = sync_pos_ / slot_size_ * slot_size_ + slot_size_ - 1;  // cannot cross one slot
    const int sync_task = sync_pos_ < sync_ceil && sync_done > sync_ceil ? sync_ceil : sync_done;
    slot = sync_task / slot_size_ * slot_size_;
    task = sync_task;
    return true;
}

bool BatchScheduler::wait_done (const int task) {
    std::unique_lock<std::mutex> taskLock (mtx_);
    while (runnable_ && done_pos_ < task)
        cv_.wait(taskLock);
    if (!runnable_)  // quit
        return false;
    return true;
}

bool BatchScheduler::wait_sync (const int task) {
    std::unique_lock<std::mutex> taskLock (mtx_);
    while (runnable_ && sync_pos_ < task)
        cv_.wait(taskLock);
    if (!runnable_)  // quit
        return false;
    return true;
}
#endif



#ifdef __CUDACC__
template <typename DT>
void TensorBuffer<DT>::page_lock () {
    cuda_check (cudaHostRegister (data_.dptr, data_.size_d(), cudaHostRegisterPortable));
    cuda_check (cudaHostRegister (pred_.dptr, pred_.size_d(), cudaHostRegisterPortable));
    cuda_check (cudaHostRegister (anno_.dptr, anno_.size_d(), cudaHostRegisterPortable));
    cuda_check (cudaHostRegister (eval_.dptr, eval_.size_d(), cudaHostRegisterPortable));
}

template <typename DT>
void TensorBuffer<DT>::page_unlk () {
    cuda_check (cudaHostUnregister (data_.dptr));
    cuda_check (cudaHostUnregister (pred_.dptr));
    cuda_check (cudaHostUnregister (anno_.dptr));
    cuda_check (cudaHostUnregister (eval_.dptr));
}
template void TensorBuffer<float >::page_lock ();
template void TensorBuffer<double>::page_lock ();
template void TensorBuffer<float >::page_unlk ();
template void TensorBuffer<double>::page_unlk ();
#else
template <typename DT>
void TensorBuffer<DT>::create (const BufferFormat& bfFormat, const Shape& src, const Shape& dst, const Shape& val, const int did) {
    bfFormat_ = bfFormat;
    did_ = did;

    list_size_ = fileData_.file_list.size();  // TODO
    slot_size_ = bfFormat_.nums * bfFormat_.numBatch;
    mini_size_ = bfFormat_.nums;

    Shape src_shape (src.rows, src.cols, src.chls, slot_size_);
    Shape dst_shape (dst.rows, dst.cols, dst.chls, slot_size_);
    Shape val_shape (val.rows, val.cols, val.chls, slot_size_);
    data_.create (src_shape, did_);
    pred_.create (dst_shape, did_);
    anno_.create (dst_shape, did_);
    eval_.create (val_shape, did_);

    scheduler_.set_size (list_size_, slot_size_, mini_size_);
}

template <typename DT>
void TensorBuffer<DT>::read_tensor (const ParaFileData& pd) {
    data_.load_bin (pd.data_path, did_);
    anno_.load_bin (pd.anno_path, did_);
    pred_.create (anno_.shape, did_);
    eval_.create (anno_.shape, did_);
    list_size_ = data_.nums();
    slot_size_ = data_.nums();
    mini_size_ = data_.nums();
}
template TensorBuffer<float >::~TensorBuffer();
template TensorBuffer<double>::~TensorBuffer();
template void TensorBuffer<float >::create (const BufferFormat& bfFormat, const Shape& src, const Shape& dst, const Shape& val, const int did);
template void TensorBuffer<double>::create (const BufferFormat& bfFormat, const Shape& src, const Shape& dst, const Shape& val, const int did);
template void TensorBuffer<float >::read_tensor (const ParaFileData& pd);
template void TensorBuffer<double>::read_tensor (const ParaFileData& pd);



template <typename DT>
void TensorBuffer<DT>::proc_image_char (const vector<unsigned char>& bytes, int& slot, int& task) {
    if (!scheduler_.proc_get (slot, task))
        return;

    BufferTrans bfTrans(bfFormat_);
    read_image_char (bfTrans, bytes, task-slot);
    scheduler_.done_add ();
}

template <typename DT>
void TensorBuffer<DT>::proc_image_file () {
    int slot, task;
    if (!scheduler_.proc_get (slot, task))
        return;

    const auto& fname = fileData_.file_list.at(task);
    name_[task-slot] = fname;

    BufferTrans bfTrans(bfFormat_);
    read_image_data (bfTrans, fileData_.data_path + fname, task-slot);
    if (fileData_.anno_type == "file") {
        const auto& fanno = fileData_.file_anno;
        if (fanno.find(fname) != fanno.end())
            read_image_anno (bfTrans, fanno.at(fname), task-slot);
    }
    else if (fileData_.anno_type == "coco_poly") {
        const auto& fanno = fileData_.coco_poly;
        if (fanno.find(fname) != fanno.end())
            read_image_anno (bfTrans, fanno.at(fname), task-slot);
    }
    else if (fileData_.anno_type == "coco_mask") {
        const auto& fanno = fileData_.coco_mask;
        if (fanno.find(fname) != fanno.end())
            read_image_anno (bfTrans, fanno.at(fname), task-slot);
    }
    scheduler_.done_add ();
}

template <typename DT>
void TensorBuffer<DT>::proc_image_file_slot (const int read_size) {
    const int proc_size = read_size > 0 ? read_size : slot_size_;
    name_.resize (proc_size);
    for (int i = 0; i < proc_size; i += mini_size_)
#pragma omp parallel for
        for (int t = 0; t < std::min (mini_size_, proc_size-i); ++t)
            proc_image_file ();
}
template void TensorBuffer<float >::proc_image_char (const vector<unsigned char>& bytes, int& slot, int& task);
template void TensorBuffer<double>::proc_image_char (const vector<unsigned char>& bytes, int& slot, int& task);
template void TensorBuffer<float >::proc_image_file ();
template void TensorBuffer<double>::proc_image_file ();
template void TensorBuffer<float >::proc_image_file_slot (const int read_size);
template void TensorBuffer<double>::proc_image_file_slot (const int read_size);



template <typename DT>
void TensorBuffer<DT>::evaluate (const int type, DT& error) {
    DT top1 = 0;
    const int chls = pred_.chls();
    const int nums = pred_.nums();
    if (type == 2) {  // segment
        for (int i = 0; i < eval_.size(); ++i)
            top1 += eval_.dptr[i];
        error += top1/nums;
    }
    else {  // classification
        for (int i = 0; i < nums; ++i) {
            DT* pptr = pred_.dptr + i * chls;
            DT* aptr = anno_.dptr + i * chls;
            DT  maxv = 0;
            int maxi = 0;
            for (int j = 0; j < chls; ++j)
                if (pptr[j] > maxv) {
                    maxv = pptr[j];
                    maxi = j;
                }
            top1 += aptr[maxi] != 1;
        }
        error += top1/nums;
    }
}
template void TensorBuffer<float >::evaluate (const int type, float& error);
template void TensorBuffer<double>::evaluate (const int type, double& error);
#endif



template <typename XPU, typename DT>
void TensorBatch<XPU, DT>::send_pred (TensorBuffer<DT>& in, const int task) const {
    const int end = task % in.slot_size_;  // TODO
    const int beg = std::max (0, end + 1 - in.mini_size_);
    in.pred_.section(beg, end).copy (pred_);
    in.eval_.section(beg, end).copy (eval_);
}

template <typename XPU, typename DT>
void TensorBatch<XPU, DT>::copy_data (const TensorBuffer<DT>& in, const int task) {
    const int end = std::max (task % in.slot_size_, in.mini_size_ - 1);
    const int beg = end + 1 - in.mini_size_;
    data_.copy (in.data_.section(beg, end));
    anno_.copy (in.anno_.section(beg, end));
}

template <typename XPU, typename DT>
void TensorBatch<XPU, DT>::rand_data (const TensorBuffer<DT>& in) {
    const int end = std::max (rand() % in.slot_size_, in.mini_size_ - 1);
    const int beg = end + 1 - in.mini_size_;
    data_.copy (in.data_.section(beg, end));
    anno_.copy (in.anno_.section(beg, end));
}
#ifdef __CUDACC__
template void TensorBatch<GPU, float>::send_pred (TensorBuffer<float>& in, const int task) const;
template void TensorBatch<GPU, float>::copy_data (const TensorBuffer<float>& in, const int task);
template void TensorBatch<GPU, float>::rand_data (const TensorBuffer<float>& in);
#else
template void TensorBatch<CPU, float>::send_pred (TensorBuffer<float>& in, const int task) const;
template void TensorBatch<CPU, float>::copy_data (const TensorBuffer<float>& in, const int task);
template void TensorBatch<CPU, float>::rand_data (const TensorBuffer<float>& in);
#endif


#endif
