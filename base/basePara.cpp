#ifndef BASE_PARA_
#define BASE_PARA_

#include "../include/base.h"

ParaModel::ParaModel (const libconfig::Config& cfg) {
    string tmp1 = cfg.lookup("model.path");  path = tmp1;
    loss_type = cfg.lookup("model.loss_type");
    if_train = cfg.lookup("model.if_train");
    if_infer = cfg.lookup("model.if_infer");
    if_update = cfg.lookup("model.if_update");
}



static void imagenet_path_list (const string& data_path, vector<string>& pathList) {
    pathList.clear ();
    vector<string> dirList = get_dir_list (data_path);
    for (auto& dir : dirList) {
        vector<string> fileList = get_file_list (dir);
        for (auto& file : fileList)
            pathList.push_back (file.substr (data_path.length()));  // imagenet和coco标注都使用相对路径
    }
}

static void imagenet_anno_list (const string& anno_path, std::unordered_map<string, int>& anno_hmap) {
    std::ifstream fstream (anno_path);
    string name;
    int anno;
    CHECK_EQ (fstream.good(), true) << "\topening failed\t" << anno_path;
    while (fstream >> name >> anno)
        anno_hmap[name] = anno;  // imagenet和coco标注都使用相对路径
}

ParaFileData::ParaFileData (const libconfig::Config& cfg, const string token) {
    string tmp0 = cfg.lookup(token+".data_type");  data_type = tmp0;
    string tmp1 = cfg.lookup(token+".data_path");  data_path = tmp1;
    string tmp2 = cfg.lookup(token+".anno_type");  anno_type = tmp2;
    string tmp3 = cfg.lookup(token+".anno_path");  anno_path = tmp3;

    if (data_type != "image")
        return;
    else
        imagenet_path_list (data_path, file_list);
    if (anno_type == "file")
        imagenet_anno_list (anno_path, file_anno);
    else if (anno_type == "coco_poly") {
        vector<COCOCImage> coco_list;
        std::unordered_map<int, vector<COCOPoly>> coco_anno;
        parse_coco_info (anno_path, coco_cats, coco_list);
        parse_coco_anno (anno_path, coco_anno);
        for (auto& image : coco_list) {
            for (auto& poly : coco_anno[image.id])
                poly.size = vector<int> {image.rows, image.cols};
            coco_poly[image.file] = coco_anno[image.id];
        }
    }
    else if (anno_type == "coco_mask") {
        std::unordered_map<int, vector<COCOMask>> coco_anno;
        parse_coco_anno (anno_path, coco_anno);
        for (auto& mask : coco_anno) {
            char image_file[32];  sprintf (image_file, "%012d.jpg", mask.first);
            coco_mask[image_file] = mask.second;
        }
    }
    else
        LOG (WARNING) << "not implemented anno type\t" << anno_type;

    LOG (INFO) << "\tdata path\t" << data_path;
    LOG (INFO) << "\tfile list\tnumFiles = " << file_list.size();
}

void ParaFileData::split_data_anno (const ParaFileData& in, const int did, const int mod) {
    data_type = in.data_type;
    data_path = in.data_path;
    anno_type = in.anno_type;
    anno_path = in.anno_path;
    coco_cats = in.coco_cats;
    for (size_t i = 0; i < in.file_list.size(); ++i)
        if (int(i) % mod == did) {
            string fname = in.file_list[i];
            file_list.emplace_back (fname);
            if (anno_type == "file" && in.file_anno.find(fname) != in.file_anno.end())
                file_anno[fname] = in.file_anno.at(fname);
            else if (anno_type == "coco_poly" && in.coco_poly.find(fname) != in.coco_poly.end())
                coco_poly[fname] = in.coco_poly.at(fname);
            else if (anno_type == "coco_mask" && in.coco_mask.find(fname) != in.coco_mask.end())
                coco_mask[fname] = in.coco_mask.at(fname);
            else
                LOG (WARNING) << "not implemented anno type\t" << anno_type;
        }
    LOG (INFO) << "\tfile list\tnumFiles = " << file_list.size();
}

#endif
