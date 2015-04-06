# vision
    learnForest.cpp 是几年前写的，现不再维护，有需要时重构
    nnet.h  神经网络头文件
    nnetConvolution.cpp 神经网络卷积层
    nnetModel.cpp 神经网络训练+预测
    optimization.h  优化算法头文件
    optimLBFGS.cpp  优化算法LBFGS
    optimVSGD.cpp   优化算法SGD
    sparse.h  稀疏矩阵头文件
    tensor.h  张量头文件
    tensorData.cpp  张量数据处理
    tensorReduce.cpp  张量归约（kernal launch部分）
    
    imagenet112_conv.cfg  imagenet数据集（128*128像素）训练的网络结构，分8层和10层两种
    imagenet224_conv.cfg  imagenet数据集（256*256像素）训练的网络结构，分8层和10层两种

深度学习工具 || 模型 || 训练 || 测试
 || 分辨率 || 模型、层数 || 每轮时间 || 轮数 || 训练时间 || 准确率
我写的 || 128*128 || 全卷积8层 || 58分钟 || 15 || 15小时 || 57.0%
 || 同上 || 10层 || 66分钟 || 15 || 17小时 || 59.6%
 || 256*256 || 全卷积8层 || 110分钟 || 15 || 28小时 || 60.5%
 || 同上 || 10层 || 200分钟 || 15 || 50小时 || 65.3%
cuda-convnet2 || 同上 || AlexNet 8层 || 约60分钟 || 90 || 90小时 || 57.7%
Caffe || 同上 || 同上 || 约80分钟 || 90 || 120小时 || 57.1%
fbcunn || 同上 || 同上 || 不详 || 53 || 不详 || 57.5%
Caffe || 同上 || GoogLeNet 22层 ||  || 60 || 约2周 || 68.7%
minerva || 同上 || 同上 ||  || 60 || *80小时 || 67.5%
