# nobodyDL
    cudaBase.cpp    CUDA基本操作
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
    tensorVML.cpp   张量向量计算
    xpu.h   设备头文件
    
    imagenet数据集配置文件（128*128像素），一些小型网络的例子
    imagenet112_conv_08.cfg   8层全卷积网络结构
        训练15轮准确率57.1%（single model single crop），等同于AlexNet 8卡并行训练90轮的准确率
        训练30轮准确率58.9%（single model single crop）

    imagenet224_hash_12.cfg  12层全卷积网络结构，256 bits 哈希学习
        训练15轮准确率63.4%（single model single crop），用GTX980ti单卡训练只需要40小时
        
    两卡训练加速1.9+倍
