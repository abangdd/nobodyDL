#ifndef UTIL_MISC_
#define UTIL_MISC_

#include <dirent.h>
#include <string.h>
#include <stdio.h>
#include <algorithm>

#include "../include/util.h"

IFileStream::IFileStream (const string path, std::ios_base::openmode mode)
  : fp_(path.c_str(), mode), path_(path)
{ CHECK_EQ (fp_.good(), true) << "\tfile open error\t" << path_;
}
IFileStream::~IFileStream()
{ fp_.close();
}

void IFileStream::read     (void *ptr, const int size)
{ fp_.read ((char*)ptr, size);
  CHECK_EQ (fp_.good(), true) << "\tfile read error\t" << fp_.eof() << fp_.fail() << fp_.bad() << "\t" << path_;
}

int IFileStream::read_size_eof ()  // from current to eof
{ const std::ios::pos_type cur_pos = fp_.tellg ();
  fp_.seekg (0,       fp_.end);
  const int size = fp_.tellg () - cur_pos;
  fp_.seekg (cur_pos, fp_.beg);
  return size;
}

unsigned char IFileStream::read_byte ()
{ unsigned char chr;
  fp_.read (reinterpret_cast<char*>(&chr), 1);  CHECK_EQ (fp_.good(), true);
  return chr;
}



OFileStream::OFileStream (const string path, std::ios_base::openmode mode)
  : fp_(path.c_str(), mode), path_(path)
{ CHECK_EQ (fp_.good(), true) << "\tfile open error\t" << path_;
}
OFileStream::~OFileStream()
{ fp_.close();
}

void OFileStream::write     (void *ptr, const int size)
{ fp_.write ((const char*)ptr, size);
  LOG_IF (ERROR, fp_.good() != true) << "\tfile write error\t" << fp_.eof() << fp_.fail() << fp_.bad() << "\t" << path_;
}



GLogHelper::GLogHelper (char* program, const char* logdir)
{ google::InitGoogleLogging (program);
  google::SetLogDestination (google::GLOG_INFO, logdir);
  google::SetStderrLogging  (google::INFO);
  google::InstallFailureSignalHandler ();
  FLAGS_colorlogtostderr = true;
  FLAGS_logbufsecs =0;
}

GLogHelper::~GLogHelper()
{ google::ShutdownGoogleLogging ();
}



void MetaImage::init (const string &a, const string &b, const string &c)
{ suffix = a;
  image_path = b;
  label_path = c;
}

void MetaImage::init (const ParaFileData &pd)
{ if (pd.type != "image")
    return;
  init ("", pd.data, pd.label);
  get_image_list ();
  get_label_list ();
  LOG (INFO) << "\timage list\tnumImages = " << imgList.size();
}

void MetaImage::init (const MetaImage &in, const int did, const int mod)
{ init (in.suffix, in.image_path, in.label_path);
  for (size_t i = 0; i < in.imgList.size(); ++i)
    if (int (i) % mod == did)
      imgList.push_back (in.imgList[i]);
  get_label_list ();
  LOG (INFO) << "\timage list\tnumImages = " << imgList.size();
}

void MetaImage::get_image_list ()
{ get_path_list (image_path, suffix, imgList);
}

void MetaImage::get_label_list ()
{ std::ifstream fp (label_path);
  CHECK_EQ (fp.good(), true) << "\topening failed\t" << label_path;

  string filename;  int label;
  while (fp >> filename >> label)
  { label_map[filename] = label;
  CHECK_EQ (fp.good(), true) << "\treading failed\t" << label_path;
  }
}



void get_dir_list (const string &dirRoot, int level, vector<string> &dirList)
{ dirList.push_back (dirRoot);
  DIR *dp = opendir (dirRoot.c_str());
  if (dp == NULL)
    LOG (FATAL) << "\tinvalid dir\t" << dirRoot;
  struct dirent* ent = NULL;
  while ((ent = readdir(dp)) != NULL)
  { const string path = dirRoot + ent->d_name + "/";
    if (ent->d_type == DT_DIR && *ent->d_name != '.')
      get_dir_list (path.c_str(), level+1, dirList);
  }
  closedir (dp);
  std::sort (dirList.begin(), dirList.end());
}

// 文件名
void get_file_list (const string &folder,  const string &suffix, vector<string> &fileList)
{ fileList.clear ();
  DIR *dp = opendir (folder.c_str());
  if (dp == NULL)
    LOG (FATAL) << "\tinvalid dir\t" << folder;
  struct dirent* ent = NULL;
  while ((ent = readdir(dp)) != NULL)
  { const string file = ent->d_name;
    if      (ent->d_type == DT_REG && suffix == "")
      fileList.push_back (file);
    else if (ent->d_type == DT_REG && file.rfind (suffix) != string::npos)
      fileList.push_back (file);
  }
  closedir (dp);
}

// 相对路径
void get_path_list (const string &dirRoot, const string &suffix, vector<string> &pathList)
{ pathList.clear ();
  vector<string> dirList;  get_dir_list  (dirRoot, -1, dirList);
  for (auto& dir : dirList)
  { vector<string> fileList;  get_file_list (dir, suffix, fileList);
    for (auto& file : fileList)
      pathList.push_back (dir.substr(dirRoot.length()) + file);
  }
}

#endif
