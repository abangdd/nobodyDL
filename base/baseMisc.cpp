#ifndef BASE_MISC_
#define BASE_MISC_

#include <algorithm>
#include <iomanip>

#include "../include/base.h"

GLogHelper::GLogHelper (char* program, const char* logdir) {
    google::InitGoogleLogging (program);
    google::SetLogDestination (google::GLOG_INFO, logdir);
    google::SetStderrLogging (google::INFO);
    google::InstallFailureSignalHandler ();
    FLAGS_colorlogtostderr = true;
    FLAGS_logbufsecs =0;
}

void SyncCV::notify () {
    std::unique_lock <std::mutex> lck (mtx_);
    while ( bval_)
        cv_.wait (lck);
    bval_ = true;
    cv_.notify_all ();
}

void SyncCV::wait () {
    std::unique_lock <std::mutex> lck (mtx_);
    while (!bval_)
        cv_.wait (lck);
    bval_ = false;
    cv_.notify_all ();
}

#endif
