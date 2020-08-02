#ifndef BASE_H_
#define BASE_H_

#include <fstream>
#include <sstream>
#include <iostream>

#include <functional>
#include <memory>
#include <regex>

#include <map>
#include <queue>
#include <unordered_map>
#include <vector>

#include <condition_variable>
#include <mutex>
#include <thread>

#include <glog/logging.h>
#include <libconfig.h++>
#include <lz4.h>
#include <mysql/mysql.h>

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

class IFileStream : private UnCopyable {
public:
    explicit IFileStream (const string path, std::ios_base::openmode mode);
    ~IFileStream () { fstream_.close(); }
    std::streamsize get_size_eof ();  // TODO
    void read (void *ptr, const int size);
    void read_lz4 (void *ptr, const int size);
private:
    std::ifstream fstream_;
    string path_;
};

class OFileStream : private UnCopyable {
public:
    explicit OFileStream (const string path, std::ios_base::openmode mode);
    ~OFileStream () { fstream_.close(); }
    void write (void *ptr, const int size);
    void write_lz4 (void *ptr, const int size);
private:
    std::ofstream fstream_;
    string path_;
};



class GLogHelper : private UnCopyable {
public:
    explicit GLogHelper (char* program, const char* logdir);
    ~GLogHelper () { google::ShutdownGoogleLogging (); }
};

class SyncCV : private UnCopyable {
public:
    explicit SyncCV () : bval_(false), ival_(0) { }
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
    explicit ParaModel () { }
    explicit ParaModel (const libconfig::Config& cfg);
public:
    int loss_type = 0;
    bool if_train;
    bool if_infer;
    bool if_update;
    string path;
};



extern std::map<int, int> coco_cat_id_map;
extern std::map<int, int> coco_voc_id_map;
extern vector<int> voc_palette;

struct COCOCategory {
    int id;
    string name;
    string supercategory;
};

struct COCOCImage {
    int id;
    int rows;
    int cols;
    string file;
};

struct COCOPoly {
    int id;
    int image_id;
    int category_id;
    float area;
    vector<int> size;  // rows cols
    vector<float> bbox;
    vector<vector<float>> polygon;
};

struct COCOMask {
    bool operator< (const COCOMask& m) { return score < m.score; }
    bool operator> (const COCOMask& m) { return score > m.score; }
    int image_id;
    int category_id;
    float score;
    vector<int> size;  // rows cols
    vector<float> bbox;
    vector<size_t> rlemask;
};

struct BoundBox {
    explicit BoundBox (const float x, const float y, const float w, const float h) : u(y-h/2), d(y+h/2), l(x-w/2), r(x+w/2) { }
    float u, d, l, r;  // 上下左右
};

class ParaFileData {
public:
    explicit ParaFileData () { }
    explicit ParaFileData (const libconfig::Config& cfg, const string token);
    void split_data_anno (const ParaFileData& in, const int did, const int mod);
public:
    string data_type, data_path;
    string anno_type, anno_path;
    vector<string> file_list;
    vector<COCOCategory> coco_cats;
    std::unordered_map<string, int> file_anno;
    std::unordered_map<string, vector<COCOPoly>> coco_poly;
    std::unordered_map<string, vector<COCOMask>> coco_mask;
};

void parse_coco_info (const string file, vector<COCOCategory>& categories, vector<COCOCImage>& images);
void parse_coco_anno (const string file, std::unordered_map<int, vector<COCOPoly>>& poly_hmap);
void parse_coco_anno (const string file, std::unordered_map<int, vector<COCOMask>>& mask_hmap);

template <typename T>
void sort_coco_anno (const vector<T>& annos, vector<T>& sorted);
template <typename T>
void hnms_coco_anno (const vector<T>& sorted, const float iou_min, vector<int>& kept);

void rle_accumulation (vector<size_t>& rle);
template <typename T>
float iou (const T& A, const T& B);



template <typename T>
class TaskPool : private UnCopyable {
public:
    bool get (T& task);
    void add (const T& task);
    void stop ();
private:
    std::queue<T> pool_;
    bool runnable_ = true;
    mutable std::mutex mtx_;
    std::condition_variable cv_;
};

class ThreadPool {
public:
    explicit ThreadPool (const size_t threads = 6);
    ~ThreadPool ();
    void run ();
private:
    TaskPool<std::function<void()>> tasks_;
    vector<std::thread> threads_;
};



vector<string> get_dir_list (const string& dirRoot);
vector<string> get_file_list (const string& folder, const std::regex& suffix = std::regex (".*"));

std::stringstream read_file (const string& path);
void save_file (const string& path, const std::stringstream& sstr, std::ios_base::openmode mode = std::ios::trunc);

void parse_row (const string& raw, const string& delim, vector<string>& cols);
void convert_row (const char** raw, const int numField, vector<string>& cols);

template <typename DT>
DT dist_l2 (const DT* x, const DT* y, const size_t cnt);
template <typename DT>
DT dist_ip (const DT* x, const DT* y, const size_t cnt);

#endif
