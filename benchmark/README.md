# imagenet分类benchmark结果
我所用的模型十分简单，只包括分组卷积、RELU、dropout、avg pool、softmax loss这五种，相当于Inception的大幅简化，故名HelloWorld  
测试过batch norm、residual、label smoothing等创新模块，对最终结果没有影响，只是loss初期下降的很快  
其他诸如drop block、drop path、mixup等数据扩充技巧因为需要训练时间翻倍，不曾测试  
![imagenet分类](20200802190159.png)

# coco分割benchmark结果
我所用的模型十分简单，只包括空洞卷积、转置卷积、RELU、sigmoid loss这四种，相当于DeepLab的大幅简化，故名HelloWorld  
GluonCV的统计口径不包括背景类，也不统计假阳，并去掉了物体小于1000像素的图片，个人无法理解，但分割结果的视觉效果与我十分相似  
Mask-RCNN因为基于检测框设计，与我存在系统性差异，除了物体外部轮廓被截断外，虽然能识别出更多小物体，但也因此存在很多假阳检测  
COCO标注质量不高，对遮挡的标注不一致、对群体的漏标十分普遍，这会造成算法识别度介于模棱两可之间，我这个模型就会产生网格状误差  
![coco分割benchmark结果](20200802190214.png)

## coco分割示例
coco原图1
![coco原图1](20200801121730.png)
coco标注1（不同颜色代表不同类别）
![coco标注1](20200801120441.png)
HelloWorld分割1（nobodyDL实现）
![HelloWorld分割1](20200801120608.png)
Mask-RCNN分割1（mmdetection实现，nobodyDL对结果做了可视化）
![Mask-RCNN分割1](20200801121555.png)



coco原图2
![coco原图2](20200801122545.png)
coco标注2（不同颜色代表不同类别）
![coco标注2](20200801122309.png)
HelloWorld分割2（nobodyDL实现）
![HelloWorld分割2](20200801122420.png)
Mask-RCNN分割2（mmdetection实现，nobodyDL对结果做了可视化）
![Mask-RCNN分割2](20200801122446.png)
