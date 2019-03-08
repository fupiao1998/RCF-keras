# RCF-keras
A keras implementation of RCF(Richer Convolutional Features for Edge Detection) for image edge detection.
Created by Yuxin Mao, if you have any problem in using it, please contact: maoyuixnssyh@my.swjtu.deu.cn
The backbone of this task now is VGG16, I will make a ResNet-based version in the future.
### To do list
- [x] VGG16 backbone
- [x] ResNet backbone
- [x] ResNeXt backbone
- [x] implementation of paper Learning to Predict Crisp Boundaries
- [ ] Evaluation function of ODS F-score
### A new implementation of paper [Learning to Predict Crisp Boundaries]!!
I implemented the edge detection model structure in the paper "Learning to Predict Crisp Boundaries"(For the convenience of description, I named it "PCB"). The main idea of the structure in the paper is to add a refinement module based on ResNeXt block, and the backbone of this paper is also VGG16. However, I have not implemented the loss function in this paper for the time being. The implementation and optimization of the loss function will be completed in the follow-up work.

What's more, I got a new idea from the "PCB" structure. Maybe I can fuse the idea of RCF and "PCB", which means combine the features of each layer of convolution in the PCB and fuse them. This experiment will also be completed in the follow-up work.
### Train
For BSDS500 dataset, you can run the file 'data_generate_forBSD.py' for generate '.npy' files for train and validation. What you need to pay attention to is to modify the path of your data set.
Then you can run 'train.py' for train RCF series models, if you want to train "PCB" model, you can run 'train_PCB.py'. The reason for the difference between the two files is that RCF combines the features of multiple convolutional layers, but the PCB uses only one.
### Model
Download the pre-trained vgg-RCF model in [BaiduNetDisk](https://pan.baidu.com/s/1IL3P8Qn-ICGYxbIFojO8XQ)
Download the pre-trained "PCB" model in [BaiduNetDisk](https://pan.baidu.com/s/1IL3P8Qn-ICGYxbIFojO8XQ)
![vgg model struct](https://github.com/fupiao1998/RCF-keras/blob/master/pictures/model%20struct.png)
### Related Projects
[RCF-pytorch](https://github.com/meteorshowers/RCF-pytorch)

[Keras-HED](https://github.com/lc82111/Keras_HED)
