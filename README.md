# vision
    learnForest.cpp 是几年前写的，现不再维护，有需要时重构
    nnet.h  神经网络头文件
    nnetBase.cpp  神经网络配置解析
    nnetConvolution.cpp 神经网络卷积层
    nnetModel.cpp 神经网络训练+预测
    optimization.h  优化算法头文件
    optimLBFGS.cpp  优化算法LBFGS
    optimVSGD.cpp   优化算法SGD
    sparse.h  稀疏矩阵头文件
    tensor.h  张量头文件
    tensorData.cpp  张量数据处理
    tensorReduce.cpp  张量归约（kernal launch部分）
    xpu.h   设备头文件
    
    imagenet112_conv.cfg  imagenet数据集（128*128像素）训练的网络结构，分 8层和10层两种
    imagenet224_conv.cfg  imagenet数据集（256*256像素）训练的网络结构，分10层和12层两种
