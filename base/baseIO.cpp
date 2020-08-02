#ifndef BASE_IO_
#define BASE_IO_

#include <algorithm>
#include <experimental/filesystem>

#include "../include/base.h"

IFileStream::IFileStream (const string path, std::ios_base::openmode mode) : fstream_(path.c_str(), mode), path_(path) {
    CHECK_EQ (fstream_.good(), true) << "\tfile open error\t" << path_;
}

std::streamsize IFileStream::get_size_eof () {
    const std::ios::streampos cur_pos = fstream_.tellg ();
    fstream_.seekg (0, fstream_.end);  // move to end 
    const std::streamsize size_eof = fstream_.tellg () - cur_pos;
    fstream_.seekg (cur_pos, fstream_.beg);  // back to beg
    return size_eof;
}

void IFileStream::read (void *ptr, const int byte_size) {
    fstream_.read ((char*)ptr, byte_size);
    CHECK_EQ (fstream_.good(), true) << "\tfile read error\t" << path_ << "\t" << fstream_.eof() << fstream_.fail() << fstream_.bad();
}

void IFileStream::read_lz4 (void *ptr, const int byte_size) {
    const int compressed_size = get_size_eof ();
    char* cptr = new char[compressed_size];
    read (cptr, compressed_size);
    const int decompressed_size = LZ4_decompress_safe (cptr, (char *)ptr, compressed_size, byte_size);
    CHECK_GE (decompressed_size, 0);
    delete[] cptr;
}



OFileStream::OFileStream (const string path, std::ios_base::openmode mode) : fstream_(path.c_str(), mode), path_(path) {
    CHECK_EQ (fstream_.good(), true) << "\tfile open error\t" << path_;
}

void OFileStream::write (void *ptr, const int byte_size) {
    fstream_.write ((const char*)ptr, byte_size);
    CHECK_EQ (fstream_.good(), true) << "\tfile write error\t" << path_ << "\t" << fstream_.eof() << fstream_.fail() << fstream_.bad();
}

void OFileStream::write_lz4 (void *ptr, const int byte_size) {
    int compressed_size = LZ4_compressBound (byte_size);
    char* cptr = new char[compressed_size];
    compressed_size = LZ4_compress ((const char *)ptr, cptr, byte_size);
    write (cptr, compressed_size);
    delete[] cptr;
}



namespace filesystem = std::experimental::filesystem;

// 根目录 子目录+/
vector<string> get_dir_list (const string& dirRoot) {
    vector<string> dirList ({dirRoot});
    for (auto& iter : filesystem::recursive_directory_iterator(dirRoot))
        if (filesystem::is_directory(iter))
            dirList.emplace_back (iter.path().string()+"/");
    return dirList;
}

// 文件名 完整路径
vector<string> get_file_list (const string& folder, const std::regex& suffix) {
    vector<string> fileList;
    for (auto& iter : filesystem::directory_iterator(folder))
        if (filesystem::is_regular_file(iter) && std::regex_match (iter.path().extension().string(), suffix))
            fileList.emplace_back (iter.path().string());
    return fileList;
}

std::stringstream read_file (const string& path) {
    std::ifstream fstream (path);  CHECK_EQ (fstream.good(), true) << "\tfile open error\t" << path;
    std::stringstream sstr;
    sstr << fstream.rdbuf();
    return sstr;
}

void save_file (const string& path, const std::stringstream& sstr, std::ios_base::openmode mode) {
    std::ofstream fstream (path, mode);  CHECK_EQ (fstream.good(), true) << "\tfile open error\t" << path;
    fstream << sstr.str();  CHECK_EQ (fstream.good(), true) << "\tfile save error\t" << path;
}

#endif
