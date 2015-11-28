# nobodyDL——小人物深度学习
	nobodyDL的理念是简单、快速、有效
	简单——nobodyDL目前只有convolution、ReLU、max/avg pooling、dropout、loss这6种网络结构
	快速——nobodyDL不希望通过牺牲速度来提高精度
	有效——nobodyDL不希望通过牺牲精度来提高速度

	在这个版本中，我提供了一个在imagenet-1k分类任务上准确率大约为71.2%的模型（224*224 central crop）
	这是之前训练60轮的结果，最新的结果是训练30轮71.1%

	这个模型有很多优点，我认为按重要程度排序分别是：
	简单的网络结构，这使他能够无缝的代替VGGNet使用到目标检测和图像分割任务中，或者方便的移植进异构平台

	Here are some prons of this model:
	a littel bit higher accuracy than other pre-trained imagenet-1k models relesaed by popular DL tools such as Caffe, mxnet
	network structure as simple as VGGNet, which makes it easier to be used in object detection and semantic segmentation tasks than BN-Inception, and easier to be migrated into other DL tools or platforms
	inference speed about 4 times faster than VGGNet 16, so it can accelerate most VGGNet based systems from quasi-realtime to realtime
	moderate model size, about 50M binary storage
	memory saving, can hold mini-batch size 128 on GTX980Ti, without using smart memory allocators
	
	and also some cons:
	no training from scratch codes provided, I am sorry for that
	I have no experiences in using other machine learning tools, so you may need to convert these model files to according formats of other tools
	
	and some explanations:
	it's unfair to compare deep learning tools while using different models, so I restrict myslef to use moderate GPU resources, moderate model size, most well-known network layers, etc
	
	and some TODOs:
	the parameter sync implementation needs to be improved before I finished building a 4 GPU system
	multi-scale training and inference
	providing some tools to migrate models into mobile DL frameworks automatically
<<<<<<< HEAD
	?ֲ?ʽ?㷨
=======
	分布式算法
>>>>>>> nobodyDL/master
	learning Markdown syntax
