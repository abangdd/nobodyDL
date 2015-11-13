# nobodyDL——小人物深度学习
	sorry it's incomplete, not ready now

	After training for 60 epochs, I got a 71.4% top 1 accuracy in the imagenet-1k classification task (validation data, 224*224 central crop), and I'am preparing on releasing this model.

	Here are some prons of this model:
	a littel bit higher accuracy than other pre-trained imagenet-1k models relesaed by popular deep learning tools such as Caffe, mxnet
	model structure as simple as VGGNet, so it is easier to use in object detection and semantic segmentation algrithims than BN-Inception
	inference speed as fast as BN-Inception, about 4 times faster than VGGNet 16, so it can accelerate most VGGNet based systems from quasi-realtime to realtime
	moderate model size, about 50M binary storage
	memory saving, I can use mini-batch size 128 on GTX980Ti, up to 256 should smart memory allocator added
	
	and also some cons:
	no DAG support, which means only simple CNN structures like VGGNet can be configured, in my opinion, whenever a basic network performance is enhanced by training algrithims, so will be complex ones
	only well-known layers are provided, they are convolution, dropout, fullconnect, ReLU, max-pooling and avg-pooling
	the training from scratch codes are not provided, so the result is not reproducible in 60 epochs
	I do not use other machine learning tools, you may need to convert the model files to according formats before using them in other tools
	
	and some explanations:
	it's unfair to compare deep learning tools when using different models, so I restrict myslef to use moderate GPU resources, moderate model size, well-known layers, etc
	
	and some TODOs:
	I found 2 dead filters (all zero) in the first layer, is that commom in other DL tools? If so, the accuracy can be further improved without more epochs
	learning Markdown syntax
