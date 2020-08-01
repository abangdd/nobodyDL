#ifndef NNET_RUN_
#define NNET_RUN_

#include <algorithm>
#include "../include/nnet.h"

template <typename XPU>
void NNetModel<XPU>::terminate () {
    for (int nid = 0; nid < para_.num_device; ++nid) {
        train_[nid].scheduler_.stop();
        infer_[nid].scheduler_.stop();
        LOG (INFO) << "\tGPU  " << nid + para_.min_device << "\tterminateed";
    }
}

template <typename XPU>
void NNetModel<XPU>::step_model (const int nid) {
    for (auto optim : optims_[nid])
        optim->ccl_update();
}



template <typename XPU>
void NNetModel<XPU>::train () {
    for (para_.now_round = para_.stt_round; para_.now_round < para_.end_round; para_.now_round++)
#pragma omp parallel for
    for (int nid = 0; nid < para_.num_device; ++nid) {
        for (auto optim : optims_[nid])
            optim->po_.set_para (para_.now_round, para_.max_round);
        train_epoch (train_[nid], batch_[nid], nid);
        valid_epoch (infer_[nid], batch_[nid], nid);
        save_model (nid);
    }
#pragma omp parallel for
    for (int nid = 0; nid < para_.num_device; ++nid) {
        valid_epoch (train_[nid], batch_[nid], nid);
        save_model (nid);
    }
}

template <typename XPU>
void NNetModel<XPU>::infer () {
#pragma omp parallel for
    for (int nid = 0; nid < para_.num_device; ++nid)
        infer_epoch (infer_[nid], batch_[nid], nid);
}



template <typename XPU>
void NNetModel<XPU>::train_epoch (TensorBuffer<float>& buffer, TensorBatch<XPU, float>& batch, const int nid) {
    const int mini_batch = buffer.bfFormat_.nums;
    const int numBatches = buffer.slot_size_ / buffer.mini_size_;
    const int numBuffers = buffer.list_size_ / buffer.slot_size_;
    std::random_shuffle (buffer.fileData_.file_list.begin(), buffer.fileData_.file_list.end());
    buffer.bfFormat_.isTrain = true;

    terr_[nid] = 0.f;
    for (int i = 0; i < numBuffers; ++i) {
        buffer.scheduler_.task_add (buffer.slot_size_);
        std::thread reader = std::thread (&TensorBuffer<float>::proc_image_file_slot, &buffer, 0);
        for (int j = 0; j < numBatches; ++j) {
            int task = (i*numBatches+j+1)*mini_batch-1;
            buffer.scheduler_.wait_done (task);
            if (para_.train_.data_type == "image")
                batch.copy_data (buffer, task);
            else
                batch.rand_data (buffer);
            fprop (nid, true);
            batch.send_pred (buffer, task);
            bprop (nid);
            step_model (nid);
          //reduce_gmat (nid);
          //bdcast_wmat (nid);
            buffer.scheduler_.sync (task);
        }
        reader.join ();
        buffer.evaluate (para_.model_.loss_type, terr_[nid]);
    }
    terr_[nid] /= numBuffers;
}

template <typename XPU>
void NNetModel<XPU>::valid_epoch (TensorBuffer<float>& buffer, TensorBatch<XPU, float>& batch, const int nid) {
    const int mini_batch = buffer.bfFormat_.nums;
    const int numBatches = buffer.slot_size_ / buffer.mini_size_;
    const int numBuffers = buffer.list_size_ / buffer.slot_size_;
    std::sort (buffer.fileData_.file_list.begin(), buffer.fileData_.file_list.end());
    buffer.bfFormat_.isTrain = false;

    perr_[nid] = 0.f;
    for (int i = 0; i < numBuffers; ++i) {
        buffer.scheduler_.task_add (buffer.slot_size_);
        std::thread reader = std::thread (&TensorBuffer<float>::proc_image_file_slot, &buffer, 0);
        for (int j = 0; j < numBatches; ++j) {
            int task = (i*numBatches+j+1)*mini_batch-1;
            buffer.scheduler_.wait_done (task);
            batch.copy_data (buffer, task);
            fprop (nid, false);
            batch.send_pred (buffer, task);
            buffer.scheduler_.sync (task);
        }
        reader.join ();
        buffer.evaluate (para_.model_.loss_type, perr_[nid]);
    }
    perr_[nid] /= numBuffers;

    char errstr[64];  sprintf (errstr, "\ttrain\t%.4f\tpredt\t%.4f", terr_[nid], perr_[nid]);
    LOG (INFO) << "\tGPU  " << nid + para_.min_device << "\tround  " << para_.now_round << errstr;
}

template <typename XPU>
void NNetModel<XPU>::infer_epoch (TensorBuffer<float>& buffer, TensorBatch<XPU, float>& batch, const int nid) {
    buffer.bfFormat_.isTrain = false;
    std::sort (buffer.fileData_.file_list.begin(), buffer.fileData_.file_list.end());
    const int list_size = buffer.list_size_;
    const int slot_size = buffer.slot_size_;
    const int mini_size = buffer.mini_size_;
    buffer.save_head_as_dict ();
    for (int i = 0; i < list_size; i += slot_size) {
        const int read_size = std::min (slot_size, list_size-i);
        buffer.scheduler_.task_add (read_size);
        std::thread reader = std::thread (&TensorBuffer<float>::proc_image_file_slot, &buffer, read_size);
        for (int j = 0; j < read_size; j += mini_size) {
            const int task = std::min (list_size, i+j+mini_size) - 1;
            buffer.scheduler_.wait_done (task);
            batch.copy_data (buffer, task);
            fprop (nid, false);
            batch.send_pred (buffer, task);
            buffer.scheduler_.sync (task);
        }
        reader.join ();
        buffer.save_pred_as_dict ();
    }
}



template <typename XPU>
void NNetModel<XPU>::infer_serve () {
#pragma omp parallel for
    for (int nid = 0; nid < para_.num_device; ++nid) {
        while (true) {  // asynchronous data copying are faster than GPU computing
          //std::this_thread::sleep_for (std::chrono::milliseconds(10));
            int slot, task;
            if (!infer_[nid].scheduler_.sync_get (slot, task))
                break;
            batch_[nid].copy_data (infer_[nid], task);
            fprop (nid, false);
            batch_[nid].send_pred (infer_[nid], task);
            infer_[nid].scheduler_.sync (task);
          //LOG (INFO) << "\tGPU  " << nid + para_.min_device << "\tinfer  " << task;
        }
    }
}

template <typename XPU>
void NNetModel<XPU>::infer_uchar (const vector<unsigned char>& bytes, const int topK, vector<NNetResult>& result) {
    const int nid = rand() % para_.num_device;
    int slot, task;
    if (!infer_[nid].scheduler_.task_add ())
        return;
    infer_[nid].proc_image_char (bytes, slot, task);
    if (!infer_[nid].scheduler_.wait_sync (task))
        return;

    TensorCPUf pred = infer_[nid].pred_[task-slot];
    result.clear();
    result.reserve (pred.size());
    const int dims = pred.size();
    for (int i = 0; i < dims; ++i)
        result.emplace_back (pred.dptr[i], i);
    std::partial_sort (result.begin(), result.begin() + std::min(topK, dims), result.end());
    result.resize (std::min(topK, dims));
}

template void NNetModel<GPU>::terminate ();
template void NNetModel<CPU>::terminate ();
template void NNetModel<GPU>::train ();
template void NNetModel<CPU>::train ();
template void NNetModel<GPU>::infer ();
template void NNetModel<CPU>::infer ();
template void NNetModel<GPU>::infer_serve ();
template void NNetModel<CPU>::infer_serve ();
template void NNetModel<GPU>::infer_uchar (const vector<unsigned char>& bytes, const int topK, vector<NNetResult>& result);
template void NNetModel<CPU>::infer_uchar (const vector<unsigned char>& bytes, const int topK, vector<NNetResult>& result);

#endif
