#ifndef UTIL_H_
#define UTIL_H_

#include <fstream>
#include <iostream>

#include <vector>
#include <unordered_map>

#include <thread>
#include <mutex>
#include <condition_variable>

#include <lz4.h>
#include <glog/logging.h>
#include <libconfig.h++>

using std::vector;
using std::string;

class UnCopyable {
protected:
  UnCopyable () { }
  ~UnCopyable() { }
private:
  UnCopyable (const UnCopyable&);
  UnCopyable& operator= (const UnCopyable&);
};

class IFileStream: private UnCopyable {
public:
  explicit IFileStream (const string path, std::ios_base::openmode mode);
  ~IFileStream();
  void read     (void *ptr, const int size);
  void read_lz4 (void *ptr, const int size);
  int read_size_eof ();  // TODO
  unsigned char read_byte ();
private:
  std::ifstream fp_;
  string path_;
};

class OFileStream: private UnCopyable {
public:
  explicit OFileStream (const string path, std::ios_base::openmode mode);
  ~OFileStream();
  void write     (void *ptr, const int size);
  void write_lz4 (void *ptr, const int size);
private:
  std::ofstream fp_;
  string path_;
};

class GLogHelper: private UnCopyable {
public:
  explicit GLogHelper (char* program, const char* logdir);
  ~GLogHelper();
};

class SyncCV: private UnCopyable {
public:
  explicit SyncCV () : bval_(false), ival_(0) { };
  void notify ();
  void wait ();
private:
  bool bval_;
  int  ival_;
  std::mutex mtx_;
  std::condition_variable cv_;
};

class ParaModel {
public:
  explicit ParaModel ();
  void set_para (const libconfig::Config &cfg);
public:
  bool if_train;
  bool if_test;
  bool if_update;
  string path;
};

class ParaDBData {
public:
  explicit ParaDBData () { };
  explicit ParaDBData (const libconfig::Config &cfg, const string token);
public:
  string path;
  int numRows;
  int threads;
  int verbose;
};

class ParaFileData {
public:
  explicit ParaFileData () { };
  explicit ParaFileData (const libconfig::Config &cfg, const string token);
public:
  string type, data, mean, eigvec, eigval, label;
};

class ParaImage {
public:
  explicit ParaImage (const int a, const int b) : rows(a), cols(b) { }
  int rows, cols;
};

class MetaImage {
public:
  void init (const string &a, const string &b, const string &c);
  void init (const ParaFileData &pd);
  void init (const MetaImage &in, const int did, const int mod);
  void get_image_list ();
  void get_label_list ();
public:
  string suffix;
  string image_path;
  string label_path;
  vector<string> imgList;
  std::unordered_map<string, int> label_map;
};

void random_index (const int cnt, vector<int> &randIdx);
void get_mem_usage ();
void get_dir_list  (const string &dirRoot, int level, vector<string> &dirList);
void get_file_list (const string &folder,  const string &suffix, vector<string> &fileList);
void get_path_list (const string &dirRoot, const string &suffix, vector<string> &pathList);
string replace (string &in, const string &subset, const string &section);

#endif
