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
using std::ios_base;
using std::ifstream;
using std::ofstream;

class IFileStream {
public:
  explicit IFileStream (const string path, ios_base::openmode mode);
  ~IFileStream();
  void read     (void *ptr, const int size);
  void read_lz4 (void *ptr, const int size);
  int read_size_eof ();  // TODO
  unsigned char read_byte ();
public:
  ifstream fp_;
  string path_;
};

class OFileStream {
public:
  explicit OFileStream (const string path, ios_base::openmode mode);
  ~OFileStream();
  void write     (void *ptr, const int size);
  void write_lz4 (void *ptr, const int size);
public:
  ofstream fp_;
  string path_;
};

class GLogHelper {
public:
  explicit GLogHelper (char* program, const char* logdir);
  ~GLogHelper();
};

class SyncCV {
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

class TimeCounter {
public:
  void start ();
  void stop ();
  float elapsed ();
private:
  struct timeval t1, t2;
};

class TimeFormatter {
public:
  void set_time ();
  void set_time (const string &hour);
  void set_second_plus (const int second);
  void set_s_e_time (const string &st, const string &et);
  string get_time ();
  string get_hour ();
  string get_day ();
public:
  struct tm time_;
  char st_[32];
  char et_[32];
  char tstr_[32];
  char hstr_[16];
  char dstr_[16];
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
  void modify_s_e_time (const TimeFormatter &tf);
  void modify_hour_tab (const TimeFormatter &tf);
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

class DataImage {
public:
  void init (const string &a, const string &b, const string &c);
  void init (const ParaFileData &pd);
  void init (const DataImage &in, const int did, const int mod);
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
