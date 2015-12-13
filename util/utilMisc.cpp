#ifndef UTIL_MISC_
#define UTIL_MISC_

#include <string.h>
#include <sys/time.h>
#include <algorithm>

#include "../include/util.h"

void SyncCV::notify ()
{ std::unique_lock <std::mutex> lck (mtx_);
  while ( bval_)
    cv_.wait (lck);
  bval_ = true;
  cv_.notify_all ();
}

void SyncCV::wait ()
{ std::unique_lock <std::mutex> lck (mtx_);
  while (!bval_)
    cv_.wait (lck);
  bval_ = false;
  cv_.notify_all ();
}


ParaModel::ParaModel () : if_train(true), if_test(true), if_update(false)
{ }

void ParaModel::set_para (const libconfig::Config &cfg)
{ if_train	= cfg.lookup ("model.if_train");
  if_test	= cfg.lookup ("model.if_test");
  if_update	= cfg.lookup ("model.if_update");
  string tmp1	= cfg.lookup ("model.path");  path = tmp1;
};

ParaDBData::ParaDBData (const libconfig::Config &cfg, const string token)
{ string tmp1	= cfg.lookup (token+".path");  path = tmp1;
  numRows	= cfg.lookup (token+".numRows");
  threads	= cfg.lookup (token+".threads");
  verbose	= cfg.lookup (token+".verbose");
};

ParaFileData::ParaFileData (const libconfig::Config &cfg, const string token)
{ string tmp0	= cfg.lookup (token+".type");   type  = tmp0;
  string tmp1	= cfg.lookup (token+".data");   data  = tmp1;
  string tmp2	= cfg.lookup (token+".label");  label = tmp2;
  string tmp3	= cfg.lookup ("statdata.mean"  );  mean   = tmp3;
  LOG (INFO) << "\tdata path\t" << data;
};


#endif
