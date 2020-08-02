# nobodyDL
个人开发的深度学习程序（部分代码），用于训练imagenet分类和coco分割，C++/CUDA实现，benchmark见![这里](https://github.com/abangdd/nobodyDL/tree/master/benchmark)  
由于缺乏机器的缘故（个人PC，硬件故障频繁），benchmark只是相对意义上的，足够work但不追求绝对意义上的SOTA  
## 已实现的功能
单机多卡（借助nccl）、rpc接口（借助brpc）、图像识别与分割、GPU batching  
## 待实现的功能
重写cudnn库（个人估计这个库中未优化的部分占了80%）、自然语言生成、图像生成  
## 不打算实现的功能
实现或提出各种创新网络结构、各种创新训练技巧  
