# nobodyDL——小人物深度学习
	sorry it's incomplete, not ready now

	After training for 60 epochs, I got a 71.4% top 1 accuracy in the imagenet-1k classification task (validation data, 224*224 central crop), and I'am preparing on releasing this model.

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
	分布式算法
	learning Markdown syntax
