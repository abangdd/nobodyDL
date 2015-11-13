# nobodyDL——小人物深度学习
	sorry it's incomplete, not ready now

	After training for 60 epochs, I got a 71.4% accuracy in the imagenet-1k classification task (validation data), and I'am preparing on releasing this model.

	Here are some prons of this model:
	a littel bit higher accuracy than other pre-trained imagenet-1k models relesaed by popular deep learning tools such as Caffe, mxnet
	model structure as simple as VGGNet, so it is easier to use in object detection and semantic segmentation algrithims than BN-Inception
	inference speed as fast as BN-Inception, about 4 times faster than VGGNet 16, so it can accelerate most VGGNet based systems from quasi-realtime to realtime
	moderate model size, about 50M binary storage
	memory saving, I can use mini-batch size 128 on GTX980Ti, up to 256 should smart memory allocator added
	
	and also some cons:
	no DAG support, which means only very simple CNN structures can be configured
	no training from scratch code provided, which means the result is not reproducible in 60 epochs
	
	and some explanations:
	it's unfair to compare deep learning tools when using different models, so I restrict myslef to use moderate GPU resources, moderate model size, well-known layers, etc
	
	and some TODOs
	I found 2 dead filters (all zero) in the first layer, is that commom in other DL tools? If so, the accuracy can be further improved without more epochs
