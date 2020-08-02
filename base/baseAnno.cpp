#ifndef BASE_ANNO_
#define BASE_ANNO_

#include <json.hpp>

#include "../include/base.h"

std::map<int, int> coco_cat_id_map {
    { 1, 0}, { 2, 1}, { 3, 2}, { 4, 3}, { 5, 4}, { 6, 5}, { 7, 6}, { 8, 7}, { 9, 8}, {10, 9},
    {11,10}, {13,11}, {14,12}, {15,13}, {16,14}, {17,15}, {18,16}, {19,17}, {20,18}, {21,19},
    {22,20}, {23,21}, {24,22}, {25,23}, {27,24}, {28,25}, {31,26}, {32,27}, {33,28}, {34,29},
    {35,30}, {36,31}, {37,32}, {38,33}, {39,34}, {40,35}, {41,36}, {42,37}, {43,38}, {44,39},
    {46,40}, {47,41}, {48,42}, {49,43}, {50,44}, {51,45}, {52,46}, {53,47}, {54,48}, {55,49},
    {56,50}, {57,51}, {58,52}, {59,53}, {60,54}, {61,55}, {62,56}, {63,57}, {64,58}, {65,59},
    {67,60}, {70,61}, {72,62}, {73,63}, {74,64}, {75,65}, {76,66}, {77,67}, {78,68}, {79,69},
    {80,70}, {81,71}, {82,72}, {84,73}, {85,74}, {86,75}, {87,76}, {88,77}, {89,78}, {90,79},
};

std::map<int, int> coco_voc_id_map {
    { 5, 0}, { 2, 1}, {16, 2}, { 9, 3}, {44, 4}, { 6, 5}, { 3, 6}, {17, 7}, {62, 8}, {21, 9},
    {67,10}, {18,11}, {19,12}, { 4,13}, { 1,14}, {64,15}, {20,16}, {63,17}, { 7,18}, {72,19}
};



void from_json (const nlohmann::json& js, COCOCategory& category) {
    js.at("id").get_to (category.id);
    js.at("name").get_to (category.name);
    js.at("supercategory").get_to (category.supercategory);
}

void from_json (const nlohmann::json& js, COCOCImage& image) {
    js.at("id").get_to (image.id);
    js.at("height").get_to (image.rows);
    js.at("width").get_to (image.cols);
    js.at("file_name").get_to (image.file);
}

void from_json (const nlohmann::json& js, COCOPoly& anno) {
    js.at("id").get_to (anno.id);
    js.at("image_id").get_to (anno.image_id);
    js.at("category_id").get_to (anno.category_id);
    js.at("area").get_to (anno.area);
    js.at("bbox").get_to (anno.bbox);
    js.at("segmentation").get_to (anno.polygon);
}

void from_json (const nlohmann::json& js, COCOMask& mask) {
    js.at("image_id").get_to (mask.image_id);
    js.at("category_id").get_to (mask.category_id);
    js.at("score").get_to (mask.score);
    js.at("bbox").get_to (mask.bbox);
    js.at("segmentation").at("size").get_to (mask.size);
    js.at("segmentation").at("counts").get_to (mask.rlemask);
}

void parse_json_file (const string file, nlohmann::json& js) {
    std::stringstream sstr = read_file (file);
    js = nlohmann::json::parse (sstr.str());
}

void parse_coco_info (const string file, vector<COCOCategory>& categories, vector<COCOCImage>& images) {
    nlohmann::json js;
    parse_json_file (file, js);

    categories = js.at("categories").get<vector<COCOCategory>>();
    images = js.at("images").get<vector<COCOCImage>>();
}

void parse_coco_anno (const string file, std::unordered_map<int, vector<COCOPoly>>& poly_hmap) {
    nlohmann::json js;
    parse_json_file (file, js);

    auto& annotations = js.at("annotations");
    for (size_t i = 0; i < annotations.size(); ++i)
        if (annotations[i].at("iscrowd").get<int>() == 0) {
            auto anno = annotations[i].get<COCOPoly>();
            auto iter = poly_hmap.find(anno.image_id);
            if (iter == poly_hmap.end())
                poly_hmap[anno.image_id] = std::move(vector<COCOPoly>{std::move(anno)});
            else
                poly_hmap[anno.image_id].emplace_back(std::move(anno));
        }
}

void parse_coco_anno (const string file, std::unordered_map<int, vector<COCOMask>>& mask_hmap) {
    nlohmann::json js;
    parse_json_file (file, js);

    for (size_t i = 0; i < js.size(); ++i)
        if (js[i].at("score").get<float>() >= 0.5) {
            auto mask = js[i].get<COCOMask>();
            auto iter = mask_hmap.find(mask.image_id);
            if (iter == mask_hmap.end())
                mask_hmap[mask.image_id] = std::move(vector<COCOMask>{std::move(mask)});
            else
                mask_hmap[mask.image_id].emplace_back(std::move(mask));
        }
}

template <typename T>
void sort_coco_anno (const vector<T>& annos, vector<T>& sorted) {
    vector<int> order (annos.size());
    std::iota (order.begin(), order.end(), 0);
    std::sort (order.begin(), order.end(), [&annos](int l, int r) { return annos[l].score > annos[r].score; });
    for (auto& i : order)
        sorted.emplace_back (annos[i]);
}
template void sort_coco_anno (const vector<COCOMask>& annos, vector<COCOMask>& sorted);

// 这里的rle是累计过的
template <typename T>
void hnms_coco_anno (const vector<T>& sorted, const float iou_min, vector<int>& kept) {
    vector<int> suppressed (sorted.size(), 0);
    for (size_t i = 0; i < sorted.size(); ++i) {
        if (suppressed[i])
            continue;
        kept.emplace_back(i);
        for (size_t j = i + 1; j < sorted.size(); ++j)
            if (!suppressed[j])
                if (iou (sorted[i], sorted[j]) > iou_min)
                    suppressed[j] = 1;
    }
}
template void hnms_coco_anno (const vector<COCOMask>& sorted, const float iou_min, vector<int>& kept);

// 这里的rle是累计过的
inline size_t rle_inter (const vector<size_t>& A, const vector<size_t>& B) {
    size_t area = 0;
    for (size_t i = 0, j = 0;;) {
        if (A.size() <= i+1 || B.size() <= j+1)
            return area;
        const size_t al = A[i], ar = A[i+1];
        const size_t bl = B[j], br = B[j+1];
        int l = std::max (al, bl);
        int r = std::min (ar, br);
        area += std::max (0, r-l);
        if (al <= bl) { i += 2;  continue; }
        if (bl <= al) { j += 2;  continue; }
    }
}

// 这里的rle是累计过的
inline size_t rle_area (const vector<size_t>& rle) {
    size_t area = 0;
    for (size_t i = 0; i < rle.size()/2; ++i)
        area += rle[i*2+1] - rle[i*2];
    return area;
}

// 这里的rle是累计过的
float iou_area (const vector<size_t>& A, const vector<size_t>& B) {
    const float inter_area = rle_inter(A, B);
    return inter_area / (rle_area(A) + rle_area(B) - inter_area + 1e-2);
}
float iou_area (const COCOMask& A, const COCOMask& B) {
    return iou_area (A.bbox, B.bbox);
}

template <typename T>
float iou (const T& A, const T& B) { return iou_area (A, B); }
template float iou (const BoundBox& A, const BoundBox& B);
template float iou (const COCOMask& A, const COCOMask& B);

#endif
