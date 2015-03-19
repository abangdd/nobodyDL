#ifndef LEARN_FOREST_
#define LEARN_FOREST_

#include "../include/util.h"
#include "../include/learning.h"

using namespace std;
using namespace tr1;



  LearnTree::LearnTree (ParaForest &pf)  // loading时要先清空
  { };

  LearnTree::LearnTree (ParaForest &pf, ParaStump &ps) :  // training时要置根节点
    pf_(&pf), ps_(&ps)
  { depths.push_back(0);
    childs.push_back(0);  // 等于0时表示叶子节点
    fids.push_back(-1);
    thrs.push_back(-1);
    tr1::shared_ptr<LearnStump> a (new LearnStump());  stumps.push_back (a);
  };

  LearnTree::~LearnTree()
  { };

  void LearnTree::get_path (const Sample &sample, int &id) const
  { id = 0;
    while (childs[id])  // 有子节点
      id = (sample.dX[fids[id]] > thrs[id]) ? childs[id] : childs[id] + 1;
  };

  void LearnTree::construct (const Sample &sample)
  { int id = 0;  get_path (sample, id);
    const int depth = depths[id];
    if (depth < pf_->maxDepth)
    { float maxFid = stumps[id]->construct (sample);
      if (maxFid >= 0)
      { const int seq = stumps.size();
        childs[id] = seq;  childs.push_back(0);  childs.push_back(0);
        depths.push_back(depth+1);  depths.push_back(depth+1);

        ModelStump stump;  stumps[id]->get_stump (maxFid, stump);
        fids[id] = stump.fid;  fids.push_back(-1);  fids.push_back(-1);
        thrs[id] = stump.thr;  thrs.push_back(-1);  thrs.push_back(-1);
        stumps.push_back (tr1::shared_ptr<LearnStump>(new LearnStump (stump.trueStat)));
        stumps.push_back (tr1::shared_ptr<LearnStump>(new LearnStump (stump.falsStat)));
        stumps[id]->deleteTest();
      }
    } else
    stumps[id]->updateStat (sample);
  };

  void LearnTree::update  (const Sample &sample) const
  { int id = 0;  get_path (sample, id);
    stumps[id]->updateStat (sample);
  };

  void LearnTree::predict (const Sample &sample, int &y) const
  { int id = 0;  get_path (sample, id);
    y = labels[id];
  };

  void LearnTree::countNodes () const
  { printf ("%lu\n", stumps.size());
  };

  void LearnTree::shrink ()
  { for (int i = 0; i < (int)stumps.size(); i++)
      stumps[i]->shrink();
  };

  void LearnTree::save (ofstream &nodeFile, ofstream &modelFile)
  { int size = stumps.size();  nodeFile.write ((const char*)&size, sizeof(int));
    nodeFile.write ((const char*)&childs[0], size * sizeof(int));
    nodeFile.write ((const char*)  &fids[0], size * sizeof(int));
    nodeFile.write ((const char*)  &thrs[0], size * sizeof(float));
    // labels在save以前都是空的
    labels.clear();
    for (int i = 0; i < size; i++)
    { VectorXf::Index maxLabel;
      stumps[i]->get_label (maxLabel);
      labels.push_back (maxLabel);
    }
    nodeFile.write ((const char*)&labels[0], size * sizeof(int));
    nodeFile.close();  modelFile.close();
  };

  void LearnTree::load (ifstream &nodeFile, ifstream &modelFile)
  { int size;  nodeFile.read ((char*)&size, sizeof(int));
    childs.resize (size);  nodeFile.read ((char*)&childs[0], size * sizeof(int));
    fids.resize (size);    nodeFile.read ((char*)  &fids[0], size * sizeof(int));
    thrs.resize (size);    nodeFile.read ((char*)  &thrs[0], size * sizeof(float));
    // labels在load以后才能用
    labels.resize (size);  nodeFile.read ((char*)&labels[0], size * sizeof(int));
    nodeFile.close();  modelFile.close();
  };



  LearnForest::LearnForest ()
  { };

  LearnForest::LearnForest (ParaData &pd, ParaForest &pf) :
    pd_(&pd), pf_(&pf), numTree(pf.numTree)
  { for (int i = 0; i < numTree; i++)
      trees.push_back (tr1::shared_ptr<LearnTree>(new LearnTree (pf)));
  };

  LearnForest::LearnForest (ParaData &pd, ParaForest &pf, ParaStump &ps) :
    pd_(&pd), pf_(&pf), ps_(&ps), numTree(pf.numTree)
  { for (int i = 0; i < numTree; i++)
      trees.push_back (tr1::shared_ptr<LearnTree>(new LearnTree (pf, ps)));
  };

  LearnForest::~LearnForest()
  { };

  void LearnForest::train (const vector<Sample> &dataset) const
  { 
#pragma omp parallel for
    for (int t = 0; t < numTree; t++)
    { vector<int> randIdx;  random_index (dataset.size(), randIdx);
      for (int s = 0; s < (int)dataset.size(); s++)
        trees[t]->construct (dataset[randIdx[s]]);
    }
  };

  void LearnForest::update (const vector<Sample> &dataset) const
  {
#pragma omp parallel for
    for (int t = 0; t < numTree; t++)
    { vector<int> randIdx;  random_index (dataset.size(), randIdx);
      for (int s = 0; s < (int)dataset.size(); s++)
        trees[t]->update (dataset[randIdx[s]]);
    }
  };

  void LearnForest::predict (const Sample &sample, Result &r) const
  { if (pf_->lossType == 0)  // 分类
    { r.dC = VectorXf::Zero (pd_->numClass);
      for (int t = 0; t < numTree; t++)  
      { int y = 0;
        trees[t]->predict (sample, y);
      	r.dC[y]++;
      }
      r.dC.array() *= 1.f / numTree;
      VectorXf::Index maxLabel;
      r.confidence = r.dC.maxCoeff (&maxLabel);  // TODO
      r.prediction = maxLabel;  // TODO
    }
  };

  void LearnForest::predict (const vector<Sample> &dataset, vector<Result> &results) const
  { results.resize (dataset.size());
    for (int i = 0; i < (int)dataset.size(); i++)
      predict (dataset[i], results[i]);
  };

  void LearnForest::countNodes () const
  { for (int i = 0; i < numTree; i++)
      trees[i]->countNodes();
  };

  void LearnForest::shrink ()
  { for (int i = 0; i < numTree; i++)
      trees[i]->shrink();
  };

  void LearnForest::save (const string path)
  { for (int i = 0; i < numTree; i++)
    { char treeid[256];  sprintf (treeid,"%d",i);
      const string fname = path+"forest_"+treeid;
      const string nodePath = fname+".node",  dataPath = fname+".data";
      ofstream nodeFile (nodePath.c_str(), ios::binary);  checkFile (nodePath, nodeFile);
      ofstream dataFile (dataPath.c_str(), ios::binary);  checkFile (dataPath, dataFile);
      trees[i]->save (nodeFile, dataFile);
    }
  };

  void LearnForest::load (const string path)
  { for (int i = 0; i < numTree; i++)
    { char treeid[256];  sprintf (treeid,"%d",i);
      const string fname = path+"forest_"+treeid;
      const string nodePath = fname+".node",  dataPath = fname+".data";
      ifstream nodeFile (nodePath.c_str(), ios::binary);  checkFile (nodePath, nodeFile);
      ifstream dataFile (dataPath.c_str(), ios::binary);  checkFile (dataPath, dataFile);
      trees[i]->load (nodeFile, dataFile);
    }
  };

#endif
