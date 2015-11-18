# nobodyDL——小人物深度学习
	sorry it's incomplete, not ready now

	After training for 60 epochs, I got a 71.4% top 1 accuracy in the imagenet-1k classification task (validation data, 224*224 central crop), and I'am preparing on releasing this model.

	Here are some prons of this model:
	a littel bit higher accuracy than other pre-trained imagenet-1k models relesaed by popular deep learning tools such as Caffe, mxnet
	model structure as simple as VGGNet, which makes it easier to be used in object detection and semantic segmentation algorithms than BN-Inception
	inference speed as fast as BN-Inception, about 4 times faster than VGGNet 16, so it can accelerate most VGGNet based systems from quasi-realtime to realtime
	moderate model size, about 50M binary storage
	memory saving, I can use mini-batch size 128 on GTX980Ti, up to 256 should smart memory allocator added
	
	and also some cons:
	no distributed learning, because I have no idea about how to design a distributed training algorithm for now
	no training from scratch codes are provided, I am sorry for that
	
	and some explanations:
	it's unfair to compare deep learning tools when using different models, so I restrict myslef to use moderate GPU resources, moderate model size, well-known layers, etc
	I have no experiences in using other machine learning tools, so you may need to convert the model files to according formats before using them in other tools
	
	and some TODOs:
	I found 2 dead filters (all zero) in the first layer, is that commom in other DL tools? If so, the accuracy can be enhanced without training more epochs
	the parameter sync implementation needs to be improved before I finished building a 4 GPU system
	learning Markdown syntax
	
	with some thoughts:
	有时候我们会看到某些经过设计的模型提升了一些准确率，我想这可能是因为这些模型的简化版还有提升的空间，在这种情况下，无论如何设计，准确率都会得到改进的，所以我不打算支持复杂的layer或者复杂的模型。
