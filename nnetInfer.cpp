#include <gflags/gflags.h>

#include "include/tensor.h"
#include "include/nnet.h"

#define XPU GPU

DEFINE_string (config, "config/infernet256_conv_50.cfg", "config file");

static std::regex suffix (".*[(.jpg)(.png)(.JPEG)]");

static void imagenet_path_list (const string& data_path, vector<string>& pathList) {
    pathList.clear ();
    vector<string> dirList = get_dir_list (data_path);
    for (auto& dir : dirList) {
        vector<string> fileList = get_file_list (dir, suffix);
        for (auto& file : fileList)
            pathList.push_back (file);
    }
}

int main (int argc, char** argv) {
    fLB::FLAGS_colorlogtostderr = true;
    google::ParseCommandLineFlags (&argc, &argv, true);
  //srand ((unsigned)time(NULL));
  //omp_set_num_threads (4);
  //mkl_set_num_threads (4);

    libconfig::Config cfg;  cfg.readFile (FLAGS_config.c_str());
    NNetModel<XPU> model;
    model.para_.config (cfg);

    dnnCtx.init (model.para_.min_device, model.para_.max_device);
    model.init_model ();
    model.init_data ();
    model.infer ();
    model.terminate ();
  //dnnCtx.release ();
}
