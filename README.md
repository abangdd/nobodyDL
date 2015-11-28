# nobodyDL——小人物深度学习
	nobodyDL的理念是简单、快速、有效
	简单——目前只有convolution、ReLU、max/avg pooling、dropout、loss这6种网络结构
	快速——不希望通过牺牲速度来提高精度
	有效——不希望通过牺牲精度来提高速度

	在这个版本中，我提供了一个在imagenet-1k分类任务上准确率大约为71.2%的模型（224*224 central crop）
	这是之前训练60轮的结果，最新的结果是训练30轮71.1%

	这个模型有很多优点，我认为按重要程度排序分别是：
	简单的网络结构，这使他能够无缝的代替VGGNet使用到目标检测和图像分割任务中，或者方便的移植进异构平台
	很高的分类精度，在开源软件中，目前是最高的
	很少的资源占用，在准确率超过70%的模型中，目前是最快的，也是显存占用最少的
	较少的模型参数，大约50兆二进制存储空间

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
